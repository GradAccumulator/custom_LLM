#include <string>
#include <map>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
using namespace std;

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME computeMV_wrapper
#endif

void compute_mv(
	torch::Tensor& m, torch::Tensor& v,
	const torch::Tensor& param,
	int t, 
	double b1, 
	double b2
) {
	m = m.to(param.device());
	v = v.to(param.device());

	m.mul_(b1).add_(param.grad().mul(1-b1));
	v.mul_(b2).add_(param.grad().pow(2).mul(1-b2));
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> bias_correction(
	torch::Tensor& m, 
	torch::Tensor& v, 
	int t,
	double b1, 
	double b2, 
	bool move_device
) {
	auto m_hat = m/(1-pow(b1, t));
	auto v_hat = v/(1-pow(b2, t));
	if(move_device){m=m.to(torch::kCPU);v=v.to(torch::kCPU);}
	return {m_hat, v_hat, m, v};
}

map<string, torch::Tensor> ComputeMV_Wrapper(
	const map<string, torch::Tensor>& params,
	map<string, torch::Tensor>& m,
	map<string, torch::Tensor>& v,
	int t,
	double b1,
	double b2,
	double eps,
	bool move_device
) {
	map<string, torch::Tensor> updated;

	for(auto& kv:params) {
		auto& m_ = m[kv.first];
		auto& v_ = v[kv.first];
		compute_mv(m_, v_, kv.second, t, b1, b2);

		const auto& [m_hat, v_hat, m_new, v_new] = bias_correction(m_, v_, t, b1, b2, move_device);

		m[kv.first] = m_new;
		v[kv.first] = v_new;

		updated[kv.first] = m_hat/(v_hat.sqrt()+eps);
	}

	return updated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("computeMV_wrapper", &ComputeMV_Wrapper);
}