#include <torch/torch.h>
#include <map>
#include <string>
using namespace std;

map<string, torch::Tensor> step_with_wd(
  map<string, map<string, torch::Tensor>>& params,
  const map<string, torch::Tensor>& gradients,
  double lr,
  double wd,
  bool use_lr_scale
) {
  for(auto& kv:params["weight_decay"]) {
    
  }
}