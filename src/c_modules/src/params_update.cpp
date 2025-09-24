#include <torch/torch.h>
#include <torch/extension.h>
#include <utility>
using namespace std;
using Any = variant<torch::Tensor, pair<torch::Tensor, double>>;

void step_with_wd(
  map<string, map<string, Any>>& params,
  map<string, torch::Tensor>& gradients,
  double lr,
  double wd
) {
  for (auto& kv:params["weight_decay"]) {
    auto* param = get_if<torch::Tensor>(&kv.second);
    param->mul_(1-lr*wd).sub_(
      gradients[kv.first], lr
    );
  }
  for (auto& kv:params["others"]) {
    get_if<torch::Tensor>(&kv.second)->sub_(
      gradients[kv.first], lr
    );
  }
  for(auto& kv: params["ls_weight_decay"]) {
    auto* pls = get_if<pair<torch::Tensor, double>>(&kv.second);
    pls->first.mul_(1-lr*pls->second*wd).sub_(
      gradients[kv.first], lr*pls->second
    );
  }
  for(auto& kv:params["ls_others"]) {
    auto* pls = get_if<pair<torch::Tensor, double>>(&kv.second);
    pls->first.sub_(
      gradients[kv.first], lr*pls->second
    );
  }
}

void step_with_none(
  map<string, map<string, Any>>& params,
  map<string, torch::Tensor>& gradients,
  double lr
) {
  
  for(auto& kv:params["others"]) {
    get_if<torch::Tensor>(&kv.second)->sub_(
      gradients[kv.first], lr
    );
  }
  for(auto& kv:params["lr_scale"]) {
    auto* pls = get_if<pair<torch::Tensor, double>>(&kv.second);
    pls->first.sub_(
      gradients[kv.first], pls->second*lr
    );
  }
}

void zero_grad(vector<torch::Tensor>& params) {
  for(auto& p:params) {
    p.grad().zero_();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("step_with_wd", &step_with_wd);
  m.def("step_with_none", &step_with_none);
  m.def("zero_grad", &zero_grad);
}