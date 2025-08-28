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
			 auto causal_mask = torch::full({ 
					 Q.size(2), Q.size(2) }, -1e8, 
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
		auto concated_heads = head_i.transpose(1, 2).reshape({batch*seq_len, d_model});
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
		auto V_i			  = saved[idx.at("V")];
		auto dOi_dVi = weight_i.transpose(2, 3).matmul(dOi_dHeadi);
		auto dOi_dWeighti = dOi_dHeadi.matmul(V_i.t());

		//V_i = X_i@Wv_i
		// x.shape					== (B, seq_len, d_model)
		//dHeadi_dVi.shape == (B, num_heads, seq_len, d_head)
		auto dO_dV = dOi_dVi.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		auto dO_dWv = x.reshape({ batch * seq_len, d_model }).t().matmul(dO_dV);
		
		//weight_i = softmax(scores)
		// scores.shape == (B, num_heads, seq_len, seq_len)
		auto scores = saved[idx.at("scores")];
		auto dSoftmaxi_dScoresi = torch::diag_embed(scores) 
			- scores.unsqueeze(-1) * scores.unsqueeze(-2);
		
		//dSoftmax_i/dScores_i.shape == (B, num_heads, seq_len, seq_len, seq_len)
		//dO_i/dWeight_i.shape == (B, num_heads, seq_len, seq_len)
		auto dOi_dScoresi = dOi_dWeighti.matmul(dSoftmaxi_dScoresi.transpose(3, 4));

		//W_ik 미분
		auto dOi_dKi = dOi_dScoresi.matmul()
	}
};
