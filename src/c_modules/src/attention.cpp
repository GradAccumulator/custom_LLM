#include <torch/torch.h>
#include <string>
#include <map>
#include <cmath>
using namespace torch::autograd;
using namespace std;

#define X 0
#define QKVWEIGHT 1
#define OUTPUTWEIGHT 2
#define SCALEQK 3
#define Q_idx 4
#define K_idx 5
#define V_idx 6
#define WEIGHT 7
#define HEADS 8
/*
ctx->save_for_backward({
	x, qkv_weight, output_weight, scale_qk,
	Q,K,V,
	atten_out[2], //weight (atten_weight)
	atten_out[3], //heads
});
*/

class CausalMaskedSelfAttetnion : public Function<CausalMaskedSelfAttetnion> {
private:
	
	//return tensor_list({Q,K,V})
	 static tensor_list get_qkv(
		torch::Tensor x,
		torch::Tensor qkv_weight,
		int num_heads,
		int d_head,
		torch::Tensor qkv_bias=torch::Tensor(),
		torch::Tensor scale_qk=torch::Tensor()
	) {
		auto QKV = x.matmul(qkv_weight);
		if (qkv_bias.defined())
			QKV = QKV + qkv_bias;
		auto QKV_reshaped = QKV.reshape({ x.sizes()[0], x.sizes()[1], num_heads, d_head *3}).transpose(1, 2);
		auto QKV_splited = QKV_reshaped.split(d_head, -1);
		if (scale_qk.defined()) 
			return {QKV_splited[0]*scale_qk, QKV_splited[1]*scale_qk, QKV_splited[2]};
		return QKV_splited;
	}

	 //return {weight}
	 static torch::Tensor softmax(
		 torch::Tensor scores,
		 int dim=-1
	 ) {
		 auto scaled_scores = scores - get<0>(scores.max(dim, true));
		 auto exps = scaled_scores.exp();
		 auto sums = exps.sum(dim, true);
		 auto weight = exps / sums;
		 return weight;
	 }

	 //return{ out,  weight, heads };
	 static tensor_list  attention(
		 AutogradContext* ctx,
		 torch::Tensor Q,
		 torch::Tensor K,
		 torch::Tensor V,
		 torch::Tensor output_weight,
		 torch::Tensor output_bias,
		 int d_head,
		 int num_heads,
		 int seq_len,
		 float p_drop,
		 torch::Tensor cached_causal_mask
		 ) {
		 auto root_d_k = sqrt(d_head);
		 auto scores = Q.matmul(K.transpose(-1, -2)) /root_d_k;
		 if (cached_causal_mask.defined() && seq_len == Q.size(2))
			 scores += cached_causal_mask;
		 else {
			 auto causal_mask = torch::full(
				 { Q.size(2), Q.size(2) }, -1e8, 
					 torch::TensorOptions().device(Q.device()).dtype(Q.dtype())
				 ).triu(1);
			 scores += causal_mask;
		 }
		 auto weight = softmax(scores, -1);
		 auto heads = weight.matmul(V);
		 //(B, Nheads, seq_len, d_head)
		 auto concated_heads = heads.transpose(1, 2).reshape({ heads.size(0), heads.size(2), num_heads * d_head });
		 auto out = concated_heads.matmul(output_weight);
		 if (output_bias.defined())
			 out += output_bias;
		 ctx->saved_data["root_d_k"] = root_d_k;
		 return{ out,  weight, heads };
	 }

public:
	static torch::Tensor forward(
		AutogradContext* ctx,
		torch::Tensor x,
		torch::Tensor qkv_weight,
		torch::Tensor output_weight,
		int num_heads,
		int d_head,
		int seq_len,
		torch::Tensor qkv_bias=torch::Tensor(),
		torch::Tensor output_bias=torch::Tensor(),
		torch::Tensor scale_qk=torch::Tensor(),
		torch::Tensor cached_causal_mask=torch::Tensor(),
		float p_drop=0.0f
	)
	{
		/*
		input shape: (B, seq_len, d_model)
		output shape: (B, seq_len, d_model)
		*/
		ctx->saved_data["num_heads"] = num_heads;
		ctx->saved_data["d_head"] = d_head;
		ctx->saved_data["use_bias"] = qkv_bias.defined();
		if (qkv_bias.defined()) {
			ctx->saved_data["qkv_bias_shape"] = qkv_bias.sizes().vec();
		}
		//ctx->saved_data["root_d_k"] = root_d_k;

		auto QKV = get_qkv(x, qkv_weight, num_heads, d_head, qkv_bias, scale_qk);
		auto Q = QKV[0];
		auto K = QKV[1];
		auto V = QKV[2];

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
		});
		return atten_out[0]; // attention output
	}

private:

public:
	static variable_list backward(
		AutogradContext* ctx,
		variable_list grad_output
	) {
		auto saved = ctx->get_saved_variables();
		auto num_heads = ctx->saved_data["num_heads"].toInt();
		auto d_head = ctx->saved_data["d_head"].toInt();
		auto root_d_k = ctx->saved_data["root_d_k"].toDouble();
		auto use_bias = ctx->saved_data["use_bias"].toBool();
		vector<int64_t> qkv_bias_shape;
		auto grad_out = grad_output[0];
		if(use_bias)qkv_bias_shape = ctx->saved_data["qkv_bias_shape"].toIntVector();
		
		torch::Tensor temp;
		variable_list temp_list;


		//x.shape == (B, seq_len, d_model)
		auto x = saved[X];
		int batch = x.size(0);
		int seq_len = x.size(1);

		auto dO_dBo = use_bias ?
			grad_out.sum(torch::OptionalIntArrayRef({ 0,1 }))
			: torch::Tensor();

		auto Wo = saved[OUTPUTWEIGHT];
		int d_model = Wo.size(1);
		auto head_i = saved[HEADS];
		//head_i.shape == (B, num_heads, seq_len, d_head)
		auto concated_heads = head_i.transpose(1, 2).reshape({ batch * seq_len, d_model });
		auto concated_grad_out = grad_out.reshape({ batch * seq_len, d_model });
		//output_i = head_i@Wo_i
		//  ㄴ미분--> dO_i/dHead_i = grad_out@Wo_i.T
		//           --> dO_i/Wo_i		 = head_i.T@grad_out
		//concated_heads.shape		== (B*seq_len, num_heads*d_head)
		// concated_grad_out.shape == (B*seq_len, d_model)
		//Wo.shape								== (num_heads*d_head, d_model)
		//grad_out.shape == (B, seq_len, d_model)
		auto dO_dWo = concated_heads.t().matmul(concated_grad_out);
		auto dO_dHead = grad_out.matmul(Wo.t());
		auto dOi_dHeadi = dO_dHead.reshape({ batch, seq_len, num_heads, d_head })
			.transpose(1, 2);

		//head_i = weight_i@V_i
		//dOutputi/dHeadi.shape == (B,num_heads,seq_len, d_head)
		//weight_i.shape			      == (B, num_heads, seq_len, seq_len)
		//V_i.shape						  == (B, num_heads, seq_len, d_head)
		auto weight_i = saved[WEIGHT];
		auto V_i = saved[V_idx];
		auto dO_dV = weight_i.transpose(2, 3).matmul(dOi_dHeadi)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		auto dOi_dWeighti = dOi_dHeadi.matmul(V_i.t());

		//Bqkv 미분용 텐서 미리 할당
		torch::Tensor dO_dBqkv;
		torch::Tensor dO_dBq;
		torch::Tensor dO_dBk;
		torch::Tensor dO_dBv;

		if (use_bias) {
			dO_dBqkv = torch::empty(
				qkv_bias_shape, 
				torch::TensorOptions().device(V_i.device())
						.dtype(V_i.dtype())
			);
			temp_list = dO_dBqkv.split(num_heads * d_head);

			dO_dBq = temp_list[0];
			dO_dBk = temp_list[1];
			dO_dBv = temp_list[2];
		}
		else {
			dO_dBqkv = torch::Tensor();
		}

		//Wqkv 미분용 텐서 미리 할당
		auto Q_i = saved[Q_idx];
		auto K_i = saved[K_idx];

		auto Wqkv = saved[QKVWEIGHT];
		auto dO_dWqkv = torch::empty_like(Wqkv);
		auto dO_dWqkv_splited = dO_dWqkv.split(num_heads * d_head, -1);

		auto dO_dWq = dO_dWqkv_splited[0];
		auto dO_dWk = dO_dWqkv_splited[1];
		auto dO_dWv = dO_dWqkv_splited[2];

		//V_i = X_i@Wv_i + Bv_i
		// x.shape					== (B, seq_len, d_model)
		//dHeadi_dVi.shape == (B, num_heads, seq_len, d_head)
		x = x.reshape({ batch * seq_len, d_model }).t();
		dO_dWv.copy_(x.matmul(dO_dV));
		if (use_bias) {
			dO_dBv.copy_(dO_dV.sum({ 0 }));
		}

		//weight_i = softmax(scores)
		// scores.shape == (B, num_heads, seq_len, seq_len)
		temp = torch::diag_embed(weight_i)
			- weight_i.unsqueeze(-1) * weight_i.unsqueeze(-2);
		//ㄴ> dSoftmax_i/dScores_i

		//W_iqkv 미분
		//dSoftmax_i/dScores_i.shape == (B, num_heads, seq_len, seq_len, seq_len)
		//dO_i/dWeight_i.shape == (B, num_heads, seq_len, seq_len)
		auto dOi_dScoresi = dOi_dWeighti.unsqueeze(3)
			.matmul(temp.transpose(3, 4))
			.squeeze(3);

		// scores    = QK.T
		// Q.shape == (B, num_heads,  seq_len, d_head)
		// K.shape == (B, num_heads,  seq_len, d_head)
		//dO_i/dScores_i.shape == (B, num_heads, seq_len, seq_len)
		//Wqkv.shape				   == (d_model, num_heads*d_head*3)
		auto dO_dQ = dOi_dScoresi.matmul(K_i)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		auto dO_dK = Q_i.matmul(dOi_dScoresi)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });

		//Q = Q*scale_qk
		//K = K*scale_qk
		//scale_qk.shape == (num_heads, 1, 1)
		//			or (1,1,1)
		auto scale_qk = saved[SCALEQK];
		torch::Tensor dO_dSqk;
		if (scale_qk.defined() && scale_qk.requires_grad()) {
			if (scale_qk.sizes() == c10::IntArrayRef({ num_heads, 1, 1 })) {
				dO_dSqk = dO_dK.reshape({batch, seq_len, d_head, num_heads})
						.sum({0, 1, 2})
					+ dO_dQ.reshape({ batch, seq_len, d_head, num_heads })
						.sum({ 0,1,2 });
			}
			else {
				dO_dSqk = dO_dK.sum() + dO_dQ.sum();
				dO_dSqk = dO_dSqk.reshape_as(scale_qk);
			}

			dO_dK *= scale_qk;
			dO_dQ *= scale_qk;
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
		//   ㄴ> (d_model, B*seq_len) 미리 바꿔놓음
		dO_dWq.copy_(x.matmul(dO_dQ));
		dO_dWk.copy_(x.matmul(dO_dK));

		//(Q/K/V) = X@W(q/k/v)
		//Wqkv.shape == (d_model, num_heads*d_head*3)
		//dO/d(Q/K).shape == (batch*seq_len, num_heads*d_head)
		//x.shape		  == (B, seq_len, d_model)
		x = x.t().reshape({ batch, seq_len, d_model });
		auto dO_dx = dO_dQ.matmul(Wqkv.slice(-1, 0, num_heads * d_head).t())
				.reshape({ batch, seq_len, d_model })
			+ dO_dK.matmul(Wqkv.slice(-1, num_heads * d_head, num_heads * d_head * 2).t())
				.reshape({ batch, seq_len, d_model })
			+ dO_dV.matmul(Wqkv.slice(-1, num_heads * d_head * 2, num_heads * d_head * 3).t())
				.reshape({ batch, seq_len, d_model });

		/*static torch::Tensor forward(
		AutogradContext* ctx,
		torch::Tensor x,
		torch::Tensor qkv_weight,
		torch::Tensor output_weight,
		int num_heads,
		int d_head,
		int seq_len,
		torch::Tensor qkv_bias=torch::Tensor(),
		torch::Tensor output_bias=torch::Tensor(),
		torch::Tensor scale_qk=torch::Tensor(),
		torch::Tensor cached_causal_mask=torch::Tensor(),
		float p_drop=0.0f
	)*/
		return {
			dO_dx,
			dO_dWqkv,
			dO_dWo,
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			dO_dBqkv,
			dO_dBo,
			dO_dSqk,
			torch::Tensor(),
			torch::Tensor(),
		};
	}
};
