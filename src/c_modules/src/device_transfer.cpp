#include <string>
#include <map>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>
using namespace std;

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME computeMV_wrapper
#endif

void MoveDevice(map<string, torch::Tensor>& parameters, torch::Device device) {
	for (auto& kv : parameters) {
		kv.second = kv.second.to(device);
	}
}

map<string, torch::Tensor> ComputeMV_Wrapper(
	const map<string, torch::Tensor>& params,
	map<string, torch::Tensor>& m,
	map<string, torch::Tensor>& v,
	int t,
	double b1,
	double b2,
	double eps,
	bool move_device,
	torch::Device device
) {
	map<string, torch::Tensor> updated;
	if (move_device) {
		MoveDevice(m, device);
		MoveDevice(v, device);
	}
	for (auto& kv : params) {
		m[kv.first] *= b1;
		v[kv.first] *= b2;

		m[kv.first] += (1-b1)*kv.second.grad();
		v[kv.first] += (1-b2)*kv.second.grad().pow(2);

		auto m_hat = m[kv.first] / (1 - pow(b1, t));
		auto v_hat = v[kv.first] / (1 - pow(b2, t));

		updated[kv.first] = m_hat / (v_hat.sqrt() + eps);
	}
	if (move_device) {
		MoveDevice(m, torch::kCPU);
		MoveDevice(v, torch::kCPU);
	}
	return updated;
}

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("computeMV_wrapper", &ComputeMV_Wrapper);
}