#include <torch/torch.h>
#include <string>
#include <map>
#include <cmath>
using namespace torch::autograd;
using namespace std;

const  map<string, int> idx = {
	{"x", 0},
	{"qkv_weight", 1},
	{"output_weight", 2},
	{"qkv_bias", 3},
	{"output_bias", 4},
	{"scale_qk", 5},
	{"Q", 6},
	{"K", 7},
	{"V", 8},
	{"scores", 9},
	{"scaled_scores", 10},
	{"weight", 11},
	{"heads", 12}
};

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

	 //return {scaled_scores, weight}
	 static tensor_list softmax(
		 torch::Tensor scores,
		 int dim=-1
	 ) {
		 auto scaled_scores = scores - get<0>(scores.max(dim, true));
		 auto exps = scaled_scores.exp();
		 auto sums = exps.sum(dim, true);
		 auto weight = exps / sums;
		 return { scaled_scores, weight };
	 }

	 //return{ out, scores, scaled_scores, weight, heads };
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
		 auto softmax_out = softmax(scores, -1);
		 auto scaled_scores = softmax_out[0]; //scaled_scores
		 auto weight = softmax_out[1]; //weight (atten_weight)
		 auto heads = weight.matmul(V);
		 //(B, Nheads, seq_len, d_head)
		 auto concated_heads = heads.transpose(1, 2).reshape({ heads.size(0), heads.size(2), num_heads * d_head });
		 auto out = concated_heads.matmul(output_weight);
		 if (output_bias.defined())
			 out += output_bias;
		 ctx->saved_data["root_d_k"] = root_d_k;
		 return{ out, scores, scaled_scores, weight, heads };
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
			x, qkv_weight, output_weight, qkv_bias, output_bias, scale_qk,
			Q,K,V,
			atten_out[1], //scores
			atten_out[2], //scaled_scores
			atten_out[3], //weight (atten_weight)
			atten_out[4], //heads
		});
		return atten_out[0]; // attention output
	}

private:

public:
	static variable_list backward(
		AutogradContext* ctx,
		variable_list grad_output
	) {
		/*ctx->save_for_backward({
			x, qkv_weight, output_weight, qkv_bias, output_bias, scale_qk,
			Q,K,V,
			atten_out[1], //scores
			atten_out[2], //scaled_scores
			atten_out[3], //weight (atten_weight)
			atten_out[4], //heads
		});*/
		auto saved = ctx->get_saved_variables();
		auto num_heads = ctx->saved_data["num_heads"].toInt();
		auto d_head = ctx->saved_data["d_head"].toInt();
		auto root_d_k = ctx->saved_data["root_d_k"].toDouble();
		auto grad_out = grad_output[0];

		/*const  map<string, int> idx = {
			{"x", 0},
			{"qkv_weight", 1},
			{"output_weight", 2},
			{"qkv_bias", 3},
			{"output_bias", 4},
			{"scale_qk", 5},
			{"Q", 6},
			{"K", 7},
			{"V", 8},
			{"scores", 9},
			{"scaled_scores", 10},
			{"weight", 11},
			{"heads", 12}
		};*/
		bool use_bias = saved[idx.at("output_bias")].defined();

		//x.shape == (B, seq_len, d_model)
		auto x = saved[idx.at("x")];
		int batch = x.size(0);
		int seq_len = x.size(1);

		auto dOut_dOutBias = use_bias ?
			grad_out.sum(torch::OptionalIntArrayRef({ 0,1 }))
			: torch::Tensor();

		auto Wo = saved[idx.at("output_weight")];
		int d_model = Wo.size(-1);
		auto head_i = saved[idx.at("heads")];
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
		auto weight_i = saved[idx.at("weight")];
		auto V_i = saved[idx.at("V")];
		auto dO_dV = weight_i.transpose(2, 3).matmul(dOi_dHeadi)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		auto dOi_dWeighti = dOi_dHeadi.matmul(V_i.t());

		//Bqkv 미분용 텐서 미리 할당
		torch::Tensor dO_dBqkv;
		torch::Tensor dO_dBqkv_splited;
		torch::Tensor dO_dBq;
		torch::Tensor dO_dBk;
		torch::Tensor dO_dBv;

		if (use_bias) {
			auto dO_dBqkv = torch::empty_like(saved[idx.at("qkv_bias")]);
			auto dO_dBqkv_splited = dO_dBqkv.split(num_heads * d_head);

			auto dO_dBq = dO_dBqkv_splited[0];
			auto dO_dBk = dO_dBqkv_splited[1];
			auto dO_dBv = dO_dBqkv_splited[2];
		}
		else {
			auto dO_dBqkv = torch::Tensor();
		}

		//Wqkv 미분용 텐서 미리 할당
		auto Q = saved[idx.at("Q")];
		auto K = saved[idx.at("K")];
		auto V = saved[idx.at("V")];

		auto qkv_weight = saved[idx.at("qkv_weight")];
		auto dO_dWqkv = torch::empty_like(qkv_weight);
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
		auto scores = saved[idx.at("scores")];
		auto dSoftmaxi_dScoresi = torch::diag_embed(scores)
			- scores.unsqueeze(-1) * scores.unsqueeze(-2);

		//W_iqkv 미분
		//dSoftmax_i/dScores_i.shape == (B, num_heads, seq_len, seq_len, seq_len)
		//dO_i/dWeight_i.shape == (B, num_heads, seq_len, seq_len)
		auto dOi_dScoresi = dOi_dWeighti.unsqueeze(3)
			.matmul(dSoftmaxi_dScoresi.transpose(3, 4))
			.squeeze(3);
		// scores    = QK.T
		// Q.shape == (B, num_heads,  seq_len, d_head)
		// K.shape == (B, num_heads,  seq_len, d_head)
		//dO_i/dScores_i.shape == (B, num_heads, seq_len, seq_len)
		//Wqkv.shape				   == (d_model, num_heads*d_head*3)
		auto dO_dQ = dOi_dScoresi.matmul(K)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		auto dO_dK = Q.matmul(dOi_dScoresi)
			.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });

		if (use_bias) {
			dO_dBq.copy_(dO_dQ.sum({ 0 }));
			dO_dBk.copy_(dO_dK.sum({ 0 }));
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
		auto Wqkv = saved[idx.at("qkv_weight")];
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

		};
	}
};
