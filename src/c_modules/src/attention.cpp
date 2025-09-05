#include <string>
#include <map>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
using namespace std;

#define X 0
#define QKV_WEIGHT 1
#define OUTPUT_WEIGHT 2
#define SCALE_QK 3
#define Q_idx 4
#define K_idx 5
#define V_idx 6
#define WEIGHT 7
#define HEADS 8
#define DROPOUT_MASK 9
#define Q_RAW 10
#define K_RAW 11

class CausalMaskedSelfAttention : public torch::autograd::Function<CausalMaskedSelfAttention> {
private:
	
	//return torch::autograd::tensor_list({Q,K,V})
	static torch::autograd::tensor_list get_qkv(
		const torch::Tensor& x,
		const torch::Tensor& qkv_weight,
		int num_heads,
		int d_head,
		const torch::Tensor& qkv_bias,
		const torch::Tensor& scale_qk
	) {
		auto QKV = x.matmul(qkv_weight);
		if (qkv_bias.numel() > 0)
			QKV = QKV + qkv_bias;
		auto QKV_reshaped = QKV.reshape({ x.sizes()[0], x.sizes()[1], num_heads, d_head *3}).transpose(1, 2);
		auto QKV_splited = QKV_reshaped.split(d_head, -1);
		if (scale_qk.numel() > 0) {
			auto temp = scale_qk;
			return {QKV_splited[0]*temp, QKV_splited[1]*temp, QKV_splited[2], QKV_splited[0], QKV_splited[1]};
		}
		return QKV_splited;
	}

	 //return {weight}
	static torch::Tensor softmax(
		const  torch::Tensor& scores,
		int dim=-1
	) {
		auto scaled_scores = scores - get<0>(scores.max(dim, true));
		auto exps = scaled_scores.exp();
		auto sums = exps.sum(dim, true);
		auto weight = exps / sums;
		return weight;
	}

	static torch::autograd::tensor_list dropout(
		torch::Tensor& x,
		float p_drop
	) {
		if (p_drop == 0.0f)return {x, torch::Tensor()};

		auto mask = ((torch::rand({x.size(-1), x.size(-2)}, torch::TensorOptions().device(x.device()).dtype(torch::kHalf)) > p_drop)/(1-p_drop)).to(x.dtype());
		return {x*mask, mask};
	}

	 //return{ out,  weight, heads, mask };
	static torch::autograd::tensor_list  attention(
		torch::autograd::AutogradContext* ctx,
		const  torch::Tensor& Q,
		const  torch::Tensor& K,
		const  torch::Tensor& V,
		const  torch::Tensor& output_weight,
		const  torch::Tensor& output_bias,
		int d_head,
		int num_heads,
		int seq_len,
		float p_drop,
		const  torch::Tensor& cached_causal_mask
		) {
		auto root_d_k = sqrt(d_head);
		auto scores = Q.matmul(K.transpose(-1, -2)) /root_d_k;
		if (cached_causal_mask.numel()>0 && seq_len == Q.size(2))
			scores += cached_causal_mask;
		else {
			auto causal_mask = torch::full(
				{ Q.size(2), Q.size(2) }, -1e8, 
					torch::TensorOptions().device(Q.device()).dtype(Q.dtype())
				).triu(1);
			scores += causal_mask;
		}
		auto weight = softmax(scores, -1);
		auto droput_out = dropout(weight, p_drop);
		auto mask = droput_out[1];
		auto weight_drop = droput_out[0];
		auto heads = weight_drop.matmul(V);
		//(B, Nheads, seq_len, d_head)
		auto concated_heads = heads.transpose(1, 2).reshape({ heads.size(0), heads.size(2), num_heads * d_head });
		auto out = concated_heads.matmul(output_weight);
		if (output_bias.numel()>0)
			out += output_bias;
		ctx->saved_data["root_d_k"] = root_d_k;
		return{ out,  weight, heads, mask};
	}

public:
	static torch::Tensor forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& x,
		const torch::Tensor& qkv_weight,
		const torch::Tensor& output_weight,
		int num_heads,
		int d_head,
		int seq_len,
		float p_drop,
		const torch::Tensor qkv_bias,
		const torch::Tensor output_bias,
		const torch::Tensor scale_qk,
		const torch::Tensor cached_causal_mask
	)
	{
		/*
		input shape: (B, seq_len, d_model)
		output shape: (B, seq_len, d_model)
		*/
		ctx->saved_data["num_heads"] = num_heads;
		ctx->saved_data["d_head"] = d_head;
		ctx->saved_data["use_bias"] = qkv_bias.numel()>0;
		if (qkv_bias.numel() > 0) {
			ctx->saved_data["qkv_bias_shape"] = qkv_bias.sizes().vec();
		}
		//ctx->saved_data["root_d_k"] = root_d_k;

		auto QKV = get_qkv(x, qkv_weight, num_heads, d_head, qkv_bias, scale_qk);
		auto Q = QKV[0];
		auto K = QKV[1];
		auto V = QKV[2];
		torch::Tensor Q_raw;
		torch::Tensor K_raw;
		if(scale_qk.numel() > 0) {
			Q_raw = QKV[3];
			K_raw = QKV[4];
		}
		else {
			Q_raw = torch::Tensor();
			K_raw = torch::Tensor();
		}

		auto atten_out = attention(
			ctx, 
			Q, K, V, 
			output_weight, 
			output_bias, 
			d_head, 
			num_heads,
			seq_len,
			p_drop,
			cached_causal_mask
		);
		//{ out, scores, scaled_scores, weight, heads}
		ctx->save_for_backward({
			x, qkv_weight, output_weight, scale_qk,
			Q,K,V,
			atten_out[1], //weight (atten_weight)
			atten_out[2], //heads
			atten_out[3], //mask
			Q_raw, K_raw
		});
		return atten_out[0]; // attention output
	}

private:
	#define DEBUG false
	#define debug if(DEBUG)cout

public:	
	static torch::autograd::variable_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::variable_list grad_output
	) {
		debug << 1 << endl;
		const auto& saved = ctx->get_saved_variables();
		auto num_heads = ctx->saved_data["num_heads"].toInt();
		auto d_head = ctx->saved_data["d_head"].toInt();
		auto root_d_k = ctx->saved_data["root_d_k"].toDouble();
		auto use_bias = ctx->saved_data["use_bias"].toBool();
		vector<int64_t> qkv_bias_shape;
		const auto& grad_out = grad_output[0];
		if(use_bias)qkv_bias_shape = ctx->saved_data["qkv_bias_shape"].toIntVector();
		
		torch::Tensor temp;
		torch::autograd::variable_list temp_list;

		//x.shape == (B, seq_len, d_model)
		debug << 2 << endl;
		int batch = saved[X].size(0);
		int seq_len = saved[X].size(1);
		
		auto dO_dBo = use_bias ?
			grad_out.sum(torch::OptionalIntArrayRef({ 0,1 }))
			: torch::Tensor();
		
		debug << 3 << endl;
		int d_model = saved[OUTPUT_WEIGHT].size(1);
		//heads.shape == (B, num_heads, seq_len, d_head)
		auto concated_heads = saved[HEADS].transpose(1, 2).reshape({ batch * seq_len, num_heads*d_head });
		auto concated_grad_out = grad_out.reshape({ batch * seq_len, d_model });
		//output_i = head_i@Wo_i
		//  ���̺�--> dO_i/dHead_i = grad_out@Wo_i.T
		//           --> dO_i/Wo_i		 = head_i.T@grad_out
		//concated_heads.shape		== (B*seq_len, num_heads*d_head)
		// concated_grad_out.shape == (B*seq_len, d_model)
		//Wo.shape								== (num_heads*d_head, d_model)
		//grad_out.shape == (B, seq_len, d_model)
		debug << 4 << endl;
		auto dO_dWo = concated_heads.t().matmul(concated_grad_out);
		auto dO_dHead = grad_out.matmul(saved[OUTPUT_WEIGHT].t());
		auto dOi_dHeadi = dO_dHead.reshape({ batch, seq_len, num_heads, d_head })
			.transpose(1, 2);

		//head_i = weight_i@V_i
		//dOutputi/dHeadi.shape == (B,num_heads,seq_len, d_head)
		//weight_i.shape			      == (B, num_heads, seq_len, seq_len)
		//V_i.shape						  == (B, num_heads, seq_len, d_head)
		debug << 5 << endl;
		torch::Tensor dO_dV;
		if (saved[DROPOUT_MASK].defined()) {
			dO_dV = (saved[WEIGHT]*saved[DROPOUT_MASK]).transpose(2, 3).matmul(dOi_dHeadi)
				.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		}
		else {
			dO_dV = (saved[WEIGHT]).transpose(2, 3).matmul(dOi_dHeadi)
				.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		}
		auto dOi_dWeighti = dOi_dHeadi.matmul(saved[V_idx].transpose(2, 3));
		if(saved[DROPOUT_MASK].defined()){
			dOi_dWeighti *= saved[DROPOUT_MASK];
		}

		//Bqkv �̺п� �ټ� �̸� �Ҵ�
		
		torch::Tensor dO_dBqkv;
		
		torch::Tensor dO_dBq;
		
		torch::Tensor dO_dBk;
		
		torch::Tensor dO_dBv;

		if (use_bias) {
			dO_dBqkv = torch::empty(
				qkv_bias_shape, 
				torch::TensorOptions().device(dO_dV.device())
						.dtype(dO_dV.dtype())
			);
			temp_list = dO_dBqkv.split(num_heads * d_head);

			dO_dBq = temp_list[0];
			dO_dBk = temp_list[1];
			dO_dBv = temp_list[2];
		}
		else {
			dO_dBqkv = torch::Tensor();
		}

		debug << 6 << endl;
		auto dO_dWqkv = torch::empty_like(saved[QKV_WEIGHT]);
		
		auto dO_dWqkv_splited = dO_dWqkv.split(num_heads * d_head, -1);
		
		auto& dO_dWq = dO_dWqkv_splited[0];
		auto& dO_dWk = dO_dWqkv_splited[1];
		auto& dO_dWv = dO_dWqkv_splited[2];

		//V_i = X_i@Wv_i + Bv_i
		
		
		debug << 7 << endl;
		auto x = saved[X].reshape({ batch * seq_len, d_model }).t();
		dO_dWv.copy_(x.matmul(dO_dV));
		if (use_bias) {
			dO_dBv.copy_(dO_dV.sum({ 0 }));
		}

		//weight_i = softmax(scores)
		//W_iqkv �̺�
		//dO_i/dWeight_i.shape == (B, num_heads, seq_len, seq_len)
		//weight_i.shape == (B, num_heads, seq_len, seq_len)
		//temp = dO_i/dScores_i
		temp = saved[WEIGHT] * dOi_dWeighti;
		if (saved[DROPOUT_MASK].defined()) {
			dOi_dWeighti *= saved[DROPOUT_MASK];
		}
		temp -= saved[WEIGHT] * temp.sum(-1, true);

		// scores    = QK.T
		// Q.shape == (B, num_heads,  seq_len, d_head)
		// K.shape == (B, num_heads,  seq_len, d_head)
		//dO_i/dScores_i.shape == (B, num_heads, seq_len, seq_len)
		//Wqkv.shape				   == (d_model, num_heads*d_head*3)
		debug << 8 << endl;
		auto dO_dQ = temp.matmul(saved[K_idx])
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head })
			/ root_d_k;
		auto dO_dK = temp.transpose(2,3).matmul(saved[Q_idx])
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head })
			/ root_d_k;

		//Q = Q*scale_qk
		//K = K*scale_qk
		//scale_qk.shape == (num_heads, 1, 1)
		//			or (1,1,1)
		//Q/K.shape == (B, num_heads, seq_len, d_head)
		debug << 9 << endl;
		torch::Tensor dO_dSqk;
		if (saved[SCALE_QK].numel()>0) {
			dO_dK = dO_dK.reshape({batch, seq_len, num_heads, d_head}).transpose(1, 2);
			dO_dQ = dO_dQ.reshape({batch, seq_len, num_heads, d_head}).transpose(1, 2);
			if(saved[SCALE_QK].requires_grad()) {
				if (saved[SCALE_QK].sizes() == c10::IntArrayRef({ num_heads, 1, 1 })) {
					dO_dSqk = (dO_dK * saved[K_RAW] + dO_dQ * saved[Q_RAW]).sum({0, 2, 3});
				}
				else {
					dO_dSqk = (dO_dK * saved[K_RAW] + dO_dQ * saved[Q_RAW]).sum();
				}
				dO_dSqk = dO_dSqk.reshape_as(saved[SCALE_QK]);
			}
			else {
				dO_dSqk = torch::Tensor();
			}
			dO_dK *= saved[SCALE_QK];
			dO_dQ *= saved[SCALE_QK];
			dO_dK = dO_dK.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
			dO_dQ = dO_dQ.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		}
		else {
			dO_dSqk = torch::Tensor();
		}
		
		if (use_bias) {
			dO_dBq.copy_(dO_dQ.sum(0));
			dO_dBk.copy_(dO_dK.sum(0));
		}

		//(Q/K) = X@(Wq/Wk)
		//dO/d(Q/K).shape == (batch*seq_len, num_heads*d_head)
		//X.shape == (B, seq_len, d_model)
		//   ��> (d_model, B*seq_len) �̸� �ٲ����
		debug << 10 << endl;
		dO_dWq.copy_(x.matmul(dO_dQ));
		dO_dWk.copy_(x.matmul(dO_dK));

		//(Q/K/V) = X@W(q/k/v)
		//Wqkv.shape == (d_model, num_heads*d_head*3)
		//dO/d(Q/K).shape == (batch*seq_len, num_heads*d_head)
		//x.shape		  == (B, seq_len, d_model)
		debug << 11 << endl;
		auto dO_dx = dO_dQ.matmul(saved[QKV_WEIGHT].slice(-1, 0, num_heads * d_head).t())
				.reshape({ batch, seq_len, d_model })
			+ dO_dK.matmul(saved[QKV_WEIGHT].slice(-1, num_heads * d_head, num_heads * d_head * 2).t())
				.reshape({ batch, seq_len, d_model })
			+ dO_dV.matmul(saved[QKV_WEIGHT].slice(-1, num_heads * d_head * 2, num_heads * d_head * 3).t())
				.reshape({ batch, seq_len, d_model });

		/*static torch::Tensor forward(
		torch::autograd::AutogradContext* ctx,
		torch::Tensor x,
		torch::Tensor qkv_weight,
		torch::Tensor output_weight,
		int num_heads,
		int d_head,
		int seq_len,
		float p_drop=0.0f,
		torch::Tensor qkv_bias=torch::Tensor(),
		torch::Tensor output_bias=torch::Tensor(),
		torch::Tensor scale_qk=torch::Tensor(),
		torch::Tensor cached_causal_mask=torch::Tensor()
	)*/
		debug << "complete" << endl;
		return {
			dO_dx,
			dO_dWqkv,
			dO_dWo,
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			dO_dBqkv,
			dO_dBo,
			dO_dSqk,
			torch::Tensor(),
		};
	}
};


torch::Tensor causal_masked_self_attention(
	const torch::Tensor& x,
	const torch::Tensor& qkv_weight,
	const torch::Tensor& output_weight,
	int num_heads,
	int d_head,
	int seq_len,
	float p_drop,
	const torch::Tensor qkv_bias,
	const torch::Tensor output_bias,
	const torch::Tensor scale_qk,
	const torch::Tensor cached_causal_mask
) {
	return CausalMaskedSelfAttention::apply(
		x,
		qkv_weight.to(x.dtype()),
		output_weight.to(x.dtype()),
		num_heads,
		d_head,
		seq_len,
		p_drop,
		qkv_bias.to(x.dtype()),
		output_bias.to(x.dtype()),
		scale_qk.to(x.dtype()),
		cached_causal_mask
	);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("causal_masked_self_attention", &causal_masked_self_attention);
}