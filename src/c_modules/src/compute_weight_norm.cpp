#include <torch/torch.h>
#include <utility>
using namespace std;

torch::Tensor compute_weight_norm(
  const vector<torch::Tensor>& params
) {
  torch::Tensor tot_norm = torch::tensor(0);
  for(auto& p: params) {
    tot_norm += p.pow(2);
  }
  return tot_norm.sqrt();
}