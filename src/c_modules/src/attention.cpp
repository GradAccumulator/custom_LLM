#include <string>
#include <map>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
using namespace std;

#define X 0
#define QKV_WEIGHT 1
#define OUTPUT_WEIGHT 2
#define QK_STD 3
#define V_IDX 4
#define WEIGHT 5
#define HEADS 6
#define DROPOUT_MASK 7
#define Q_RAW 8
#define K_RAW 9
#define Q_SCALE 10
#define K_SCALE 11
#define CLIP_MASK 12
#define A_H 13
#define WEIGHT_DROP 14
#define SAVED_LENGTH 15

class CausalMaskedSelfAttention : public torch::autograd::Function<CausalMaskedSelfAttention> {
private:
	#define DEBUG false
	#define debug(...) do {if(DEBUG) cout << __VA_ARGS__ << endl;} while(0)
	
	//Returns {Q, K, V} or {Q, K, V, Q_raw, K_raw, Q_scale, K_scale, qk_std}
	static torch::autograd::tensor_list get_qkv(
		const torch::Tensor& x,
		const torch::Tensor& qkv_weight,
		int num_heads,
		int d_head,
		const torch::Tensor& qkv_bias,
		const torch::Tensor& s_h,
		const double& eps,
		torch::Tensor& k_cache,
		torch::Tensor& v_cache,
		bool is_inference,
		int& cache_offset//,
		//int seq_end
	) {
		//W_qkv를 이용하여 Q,K,V를 한번에 구한 후 
		//B×H×L×d_head로 바꾼 뒤, .split()을 통해서 Q,K,V를 분리
		//또한 bias_qkv가 있을 시 추가로 더함
		debug("get_qkv in");
		auto QKV = x.matmul(qkv_weight);//x.shape == (B, L, d), qkv_weight.shape == (d, H*d_head*3)
		//-> QKV.shape == (B, L, H*d_head*3)
		if (qkv_bias.numel() > 0)
			QKV.add_(qkv_bias);//qkv_bias.shape == (H*d_head*3)

		debug("get_qkv" << 1);
		auto QKV_splited = QKV.split(num_heads*d_head, -1);
		auto Q = QKV_splited[0].reshape({x.sizes()[0], x.sizes()[1], num_heads, d_head}).transpose(1, 2);
		auto K = QKV_splited[1].reshape({x.sizes()[0], x.sizes()[1], num_heads, d_head}).transpose(1, 2);
		auto V = QKV_splited[2].reshape({x.sizes()[0], x.sizes()[1], num_heads, d_head}).transpose(1, 2);
		if(is_inference) {
			Q = Q.slice(1, 0, 1);
			//k_cache.slice(1, cache_offset, cache_offset+seq_end).copy_(K.slice(1, 0, seq_end));
			//v_cache.slice(1, cache_offset, cache_offset+seq_end).copy_(V.slice(1, 0, seq_end));
			//cache_offset += seq_end; 
			K = k_cache;
			V = v_cache;
		}

		debug("get_qkv" << 2);
		//qk_std를 사용할 때에만 실행
		if (s_h.numel() > 0) {
			//fp16연산에서는 ϵ의 값을 1e-4로, 이외에는 1e-7로 설정
			auto log_eps = (s_h.dtype() == torch::kHalf) ? 1e-4f : 1e-7f;

			//a_h>ε 여부 저장
			auto clip_mask = s_h>log_eps;
			//전달받은 s_h(=a)를 이용하여 log값(=s_h)을 구함
			auto qk_std = (s_h+1).clamp_min(1+log_eps).log();

			//Q_h와 K_h의 표준편차를 구함
			//Q/K.shape == (B, H, L, d_head)
			//std over L,d_head
			auto std_Q = Q.std(c10::OptionalIntArrayRef({2, 3}), /*unbiased=*/false, /*keepdim=*/true);
			auto std_K = K.std(c10::OptionalIntArrayRef({2, 3}), /*unbiased=*/false, /*keepdim=*/true);
			//-> std_Q/K.shape == (B, H, 1, 1)

			//Q_h, K_h의 표준편차와 qk_std를 이용하여 Q_h와 K_h의 표준편차를 고정
			//qk_std.shape == (H, 1, 1)
			//std_Q/K_i.shape == (B, H, 1, 1)
			auto inv_std_q = 1/(std_Q + eps);
			auto inv_std_k = 1/(std_K + eps);
			auto Q_scale = qk_std.mul(inv_std_q);
			auto K_scale = qk_std.mul(inv_std_k);
			//-> Q/K_scale.shape == (B, H, 1, 1)
			//**broadcasted**
			debug("get_qkv end");
			//backward에 필요한 중간값과 결과값을 반환
			return {
					Q*Q_scale, K*K_scale, V, 
					Q, K, 
					Q_scale, K_scale, qk_std,
					clip_mask,
				};
		}

		//qk_std를 사용하지 않을 때에는 Q,K,V를 그대로 반환
		return {Q, K, V};
	}

	 //return {weight}
	static torch::Tensor softmax(
		const  torch::Tensor& scores,
		int dim=-1
	) {
		auto scaled_scores = scores - get<0>(scores.max(dim, true));
		auto exps = scaled_scores.exp();
		return exps.div_(exps.sum(dim, true));
	}

	static torch::autograd::tensor_list dropout(
		torch::Tensor& x,
		float p_drop
	) {
		debug("dropout in");
		if (p_drop == 0.0f)return {x, torch::Tensor()};

		auto mask = (torch::rand({x.size(-2), x.size(-1)}, torch::TensorOptions().device(x.device()).dtype(torch::kHalf)) > p_drop).to(x.dtype()) / (1-p_drop);
		debug("dropout end");
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
		debug("attention in");
		auto root_d_k = sqrt(d_head);
		auto scores = Q.matmul(K.transpose(-1, -2)) /root_d_k;
		if (cached_causal_mask.numel()>0 && seq_len == Q.size(2))
			scores.add_(cached_causal_mask);
		else {
			auto causal_mask = torch::full(
				{ Q.size(2), Q.size(2) }, -1e4, 
					torch::TensorOptions().device(Q.device()).dtype(Q.dtype())
				).triu(1);
			scores.add_(causal_mask);
		}
		debug("attention " << 1);
		auto weight = softmax(scores, -1);
		auto droput_out = dropout(weight, p_drop);
		auto heads = droput_out[0].matmul(V);
		//(B, Nheads, seq_len, d_head)
		auto concated_heads = heads.transpose(1, 2).reshape({ heads.size(0), heads.size(2), num_heads * d_head });
		auto out = concated_heads.matmul(output_weight);
		if (output_bias.numel()>0)
			out.add_(output_bias);
		ctx->saved_data["root_d_k"] = root_d_k;
		debug("attention end");
		return{ out, weight, heads, /*mask=*/droput_out[1], /*weight(drop)*/droput_out[0]};
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
		const torch::Tensor a_h,
		const torch::Tensor cached_causal_mask,
		const float& eps,
		torch::Tensor& k_cache,
		torch::Tensor& v_cache,
		bool is_inference,
		int& cache_offset//,
		//int seq_end
	)
	{
		debug("forward in");
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
		ctx->saved_data["s_h_is_learnable"] = a_h.requires_grad();
		//ctx->saved_data["root_d_k"] = root_d_k;
		debug("forward" << 1);
		auto QKV = get_qkv(x, qkv_weight, num_heads, d_head, qkv_bias, a_h, eps, k_cache, v_cache, is_inference, cache_offset/*, seq_end*/);
		auto Q = QKV[0];
		auto K = QKV[1];
		auto V = QKV[2];
		torch::Tensor Q_raw;
		torch::Tensor K_raw;
		torch::Tensor Q_scale;
		torch::Tensor K_scale;
		torch::Tensor qk_std;
		torch::Tensor clip_mask;
		if(a_h.numel() > 0) {
			Q_raw = QKV[3];
			K_raw = QKV[4];
			Q_scale = QKV[5];
			K_scale = QKV[6];
			qk_std = QKV[7];
			clip_mask = QKV[8];
		}
		debug("forward" << 2);
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
		debug("forward, save for backward");
		//{ out, scores, scaled_scores, weight, heads}
		ctx->saved_data["dtype"] = x.dtype().toScalarType();
		if(!Q_raw.defined()) {
			Q_raw = Q;
			K_raw = K;
		}
		ctx->save_for_backward({
			x									, 
			qkv_weight				, 
			output_weight			, 
			qk_std						,
			V									,
			atten_out[1]			, //weight (atten_weight)
			atten_out[2]			, //heads
			atten_out[3]			, //mask
			Q_raw							, 
			K_raw							,
			Q_scale						, 
			K_scale						,
			clip_mask					,
			a_h								,
			atten_out[4]			//weight(drop)
		});
		return atten_out[0]; // attention output
	}

public:	
	static torch::autograd::variable_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::variable_list grad_output
	) {
		debug("backward" << 0);
		auto saved = ctx->get_saved_variables();
		auto num_heads = ctx->saved_data["num_heads"].toInt();
		auto d_head = ctx->saved_data["d_head"].toInt();
		auto root_d_k = ctx->saved_data["root_d_k"].toDouble();
		auto use_bias = ctx->saved_data["use_bias"].toBool();
		auto s_h_is_learnable = ctx->saved_data["s_h_is_learnable"].toBool();
		vector<int64_t> qkv_bias_shape;
		const auto& grad_out = grad_output[0];
		if(use_bias)qkv_bias_shape = ctx->saved_data["qkv_bias_shape"].toIntVector();
		
		torch::Tensor temp;
		torch::autograd::variable_list temp_list;
		
		auto dtype = ctx->saved_data["dtype"].toScalarType();
		for(int i=0;i<SAVED_LENGTH;i++)
			if(saved[i].numel() > 0)
				saved[i] = saved[i].to(dtype);

		//x.shape == (B, seq_len, d_model)
		int batch = saved[X].size(0);
		int seq_len = saved[X].size(1);
		
		auto dO_dBo = use_bias ?
			grad_out.sum(torch::OptionalIntArrayRef({ 0,1 }))
			: torch::Tensor();
		
		debug("backward" << 1);
		int d_model = saved[OUTPUT_WEIGHT].size(1);
		//heads.shape == (B, num_heads, seq_len, d_head)
		auto concated_heads = saved[HEADS].transpose(1, 2).reshape({batch,seq_len,num_heads*d_head})
		.reshape({ batch * seq_len, num_heads * d_head });
		auto concated_grad_out = grad_out.reshape({ batch * seq_len, d_model });
		//output_i = head_i@Wo_i
		//  ���̺�--> dO_i/dHead_i = grad_out@Wo_i.T
		//           --> dO_i/Wo_i		 = head_i.T@grad_out
		//concated_heads.shape		== (B*seq_len, num_heads*d_head)
		// concated_grad_out.shape == (B*seq_len, d_model)
		//Wo.shape								== (num_heads*d_head, d_model)
		//grad_out.shape == (B, seq_len, d_model)
		auto dO_dWo = concated_heads.t().matmul(concated_grad_out);
		auto dO_dHead = grad_out.matmul(saved[OUTPUT_WEIGHT].t());
		auto dOi_dHeadi = dO_dHead.reshape({ batch, seq_len, num_heads, d_head })
			.transpose(1, 2);

		//head_i = weight_i@V_i
		//dOutputi/dHeadi.shape == (B,num_heads,seq_len, d_head)
		//weight_i.shape			      == (B, num_heads, seq_len, seq_len)
		//V_i.shape						  == (B, num_heads, seq_len, d_head)
		torch::Tensor dO_dV;
		if (saved[DROPOUT_MASK].defined()) {
			dO_dV = (saved[WEIGHT_DROP]).transpose(2, 3).matmul(dOi_dHeadi)
				.transpose(1, 2).reshape({ batch, seq_len, num_heads * d_head }).reshape({ batch * seq_len, num_heads * d_head });
		}
		else {
			dO_dV = (saved[WEIGHT]).transpose(2, 3).matmul(dOi_dHeadi)
				.transpose(1, 2).reshape({ batch, seq_len, num_heads * d_head }).reshape({ batch * seq_len, num_heads * d_head });
		}
		auto dOi_dWeighti = dOi_dHeadi.matmul(saved[V_IDX].transpose(2, 3));
		if(saved[DROPOUT_MASK].defined()){
			dOi_dWeighti.mul_(saved[DROPOUT_MASK]);
		}

		//Bqkv �̺п� �ټ� �̸� �Ҵ�
		
		torch::Tensor dO_dBqkv;
		torch::Tensor dO_dBq;
		torch::Tensor dO_dBk;
		torch::Tensor dO_dBv;

		debug("backward" << 2);
		if (use_bias) {
			dO_dBqkv = torch::empty(
				qkv_bias_shape, 
				torch::TensorOptions().device(dO_dV.device())
						.dtype(dO_dV.dtype())
			);
			temp_list = dO_dBqkv.split(num_heads * d_head, 0);

			dO_dBq = temp_list[0];
			dO_dBk = temp_list[1];
			dO_dBv = temp_list[2];
		}
		else {
			dO_dBqkv = torch::Tensor();
		}

		auto dO_dWqkv = torch::empty_like(saved[QKV_WEIGHT]);//(d_model, num_heads*d_head*3)
		
		auto dO_dWqkv_splited = dO_dWqkv.split(num_heads * d_head, -1);
		
		auto& dO_dWq = dO_dWqkv_splited[0];
		auto& dO_dWk = dO_dWqkv_splited[1];
		auto& dO_dWv = dO_dWqkv_splited[2];

		//V_i = X_i@Wv_i + Bv_i
		
		debug("backward" << 3);
		auto x = saved[X].reshape({ batch * seq_len, d_model }).t();
		dO_dWv.copy_(x.matmul(dO_dV));

		//weight_i = softmax(scores)
		//W_iqkv �̺�
		//dO_i/dWeight_i.shape == (B, num_heads, seq_len, seq_len)
		//weight_i.shape == (B, num_heads, seq_len, seq_len)
		//temp = dO_i/dScores_i
		//temp = saved[WEIGHT] * dOi_dWeighti;
		//temp -= saved[WEIGHT] * temp.sum(-1, true);
		temp = saved[WEIGHT].mul(dOi_dWeighti);
		temp.sub_(saved[WEIGHT].mul(temp.sum(-1, true)));

		// scores    = QK.T
		// Q.shape == (B, num_heads,  seq_len, d_head)
		// K.shape == (B, num_heads,  seq_len, d_head)
		//dO_i/dScores_i.shape == (B, num_heads, seq_len, seq_len)
		//Wqkv.shape				   == (d_model, num_heads*d_head*3)
		torch::Tensor dO_dK;
		torch::Tensor dO_dQ;
		if(saved[QK_STD].defined() && saved[QK_STD].numel() > 0){
			dO_dQ = temp.matmul(saved[K_RAW].mul(saved[K_SCALE]))
				.transpose(1, 2).reshape({ batch,seq_len, num_heads * d_head })
				.view({ batch * seq_len, num_heads * d_head })
				.div(root_d_k);
			dO_dK = temp.transpose(2,3).matmul(saved[Q_RAW].mul(saved[Q_SCALE]))
				.transpose(1, 2).reshape({ batch,seq_len, num_heads * d_head })
				.view({ batch * seq_len, num_heads * d_head })
				.div(root_d_k);
		}
		else {
			dO_dQ = temp.matmul(saved[K_RAW])
				.transpose(1, 2).reshape({ batch,seq_len, num_heads * d_head })
				.view({ batch * seq_len, num_heads * d_head });
			dO_dK = temp.transpose(2, 3).matmul(saved[Q_RAW])
				.transpose(1, 2).reshape({ batch,seq_len, num_heads * d_head })
				.view({ batch * seq_len, num_heads * d_head });
		}

		//Q = Q*s_h
		//K = K*s_h
		//s_h.shape == (num_heads, 1, 1)
		//			or (1,1,1)
		//Q/K.shape == (B, num_heads, seq_len, d_head)
		//dO_dAh 미리 할당
		debug("backward" << 4);
		torch::Tensor dO_dAh;
		//QK_STD가 있을때만 실행
		if (saved[QK_STD].numel()>0 && saved[QK_STD].defined()) {
			//dO_dK/Q.shape == (B, seq_len, num_heads*d_head)
			//-> (B, seq_len, num_heads, d_head)
			//-> (B, num_heads, seq_len, d_head)
			//dO_dK/Q_h 형태로 변환
			dO_dQ/*_raw*/ = dO_dQ.reshape({batch, seq_len, num_heads, d_head}).transpose(1, 2);
			dO_dK/*_raw*/ = dO_dK.reshape({batch, seq_len, num_heads, d_head}).transpose(1, 2);

			//s_h(=a_h)가 learnable일 경우에만 실행
			if(s_h_is_learnable){
				auto& inv_std_q = saved[Q_SCALE].mul_(saved[QK_STD]);
				auto& inv_std_k = saved[K_SCALE].mul_(saved[QK_STD]);
				//saved[Q/K_raw] = Q/K_raw
				//inv_std_q/k = 1/(σ_q/k+ϵ)
				//미분식 그대로 계산
				//∑_(b=1)^B x ∑_(t=1)^L ∑_(m=1)^(d_head) x (dL/(dQ_b,h,t,m)∘(dQ_b,h,t,m^(raw))/(ds_(h)+dL/(dK_b,h,t,m)∘(dK_b,h,t,m^(raw))/(ds_h))
				dO_dAh = dO_dQ/*_raw*/.mul(saved[Q_RAW]).mul_(inv_std_q)
					.add_(dO_dK/*_raw*/.mul(saved[K_RAW]).mul_(inv_std_k))
					.sum({0, 2, 3});
				
				//  (dO_dK * saved[K_RAW]*inv_std_k
				// 	+ dO_dQ * saved[Q_RAW]*inv_std_q)
				// 		.sum({0, 2, 3});
				//-> 이때 dO_dAh.shape == (num_heads,)

				//SqkExp == a_h
				//dO_dAh.shape == (num_heads,)
				//-> dO_dAh.shape == (num_heads, 1, 1) 변환
				//ds_h/da_h== | 1/a_h (a_h>eps)
				//           | 0    (a_h<=eps)
				dO_dAh = dO_dAh.reshape_as(saved[QK_STD])
					.div_(saved[A_H])// 1/a_h = 1/exp(s_h)
					.mul_(saved[CLIP_MASK]); // CLIP_MASK -> 0(클립 된 곳) or 1(클립 안 된 곳)
			}
			else {
				//a_h가 learnable이 아닐 때는 빈 텐서로 메꿈
				dO_dAh = torch::Tensor();
			}
			//forward => Q = Q_raw * Q_scale, K = K_raw * K_scale
			//dO_dQ/k = dO_dQ_raw * Q_scale
			dO_dQ.mul_(saved[Q_SCALE]);
			dO_dK.mul_(saved[K_SCALE]);

			//기울기 계산 이후 dO_dQ/K_h 형태에서 dO_dQ/K로 변환
			dO_dQ = dO_dQ.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
			dO_dK = dO_dK.transpose(1, 2).reshape({ batch * seq_len, num_heads * d_head });
		}
		else {
			//a_h를 사용하지 않을 때는 빈 텐서로 메꿈
			dO_dAh = torch::Tensor();
		}
		
		if (use_bias) {
			dO_dBq.copy_(dO_dQ.sum(0));
			dO_dBk.copy_(dO_dK.sum(0));
			dO_dBv.copy_(dO_dV.sum(0));
		}

		debug("backward" << 5);
		//(Q/K) = X@(Wq/Wk)
		//dO/d(Q/K).shape == (batch*seq_len, num_heads*d_head)
		//X.shape == (B, seq_len, d_model)
		//   ��> (d_model, B*seq_len) �̸� �ٲ����
		dO_dWq.copy_(x.matmul(dO_dQ));
		dO_dWk.copy_(x.matmul(dO_dK));

		//(Q/K/V) = X@W(q/k/v)
		//Wqkv.shape == (d_model, num_heads*d_head*3)
		//dO/d(Q/K).shape == (batch*seq_len, num_heads*d_head)
		//x.shape		  == (B, seq_len, d_model)
		auto Wqkv = saved[QKV_WEIGHT].split(num_heads * d_head, -1);
		auto Wq=Wqkv[0], Wk=Wqkv[1], Wv=Wqkv[2];

		auto dO_dx =  dO_dQ.matmul(Wq.t()).reshape({ batch, seq_len, d_model })
								.add_(dO_dK.matmul(Wk.t()).reshape({ batch, seq_len, d_model }))
								.add_(dO_dV.matmul(Wv.t()).reshape({ batch, seq_len, d_model }));

		/*static torch::Tensor forward(
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
		const torch::Tensor s_h,
		const torch::Tensor cached_causal_mask,
		const float& eps,
		torch::Tensor& k_cache,
		torch::Tensor& v_cache,
		bool is_inference,
		int& cache_offset,
		int seq_end
	)*/
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
			dO_dAh,
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor(),
			torch::Tensor()//,
			//torch::Tensor()
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
	const torch::Tensor& qkv_bias,
	const torch::Tensor& output_bias,
	const torch::Tensor& s_h,
	const torch::Tensor& cached_causal_mask,
	const float& eps,
	torch::Tensor& k_cache,
	torch::Tensor& v_cache,
	bool is_inference,
	int& cache_offset//,
	//int seq_end
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
		s_h.to(x.dtype()),
		cached_causal_mask.to(x.dtype()),
		eps,
		k_cache,
		v_cache,
		is_inference,
		cache_offset//,
		//seq_end
	);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("causal_masked_self_attention", &causal_masked_self_attention);
}