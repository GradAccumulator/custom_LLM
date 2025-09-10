#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

bool grad_nan_check(torch::autograd::tensor_list list) {
  for (auto& t : list) {
    if (torch::isnan(t.grad()).any().item<bool>()) return true;
  }
  return false;
}
bool nan_check(torch::TensorList list) {
  for (auto& t : list) {
    if (torch::isnan(t).any().item<bool>()) return true;
  }
  return false;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("grad_nan_check", &grad_nan_check);
  m.def("nan_check", &nan_check);
}