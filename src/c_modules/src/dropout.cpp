#include <torch/torch.h>
#include <torch/extension.h>
#include <utility>
using namespace std;

class Dropout : public torch::autograd::Function<Dropout> {
private:
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor& x,
    float p
  ) { 
    ctx->saved_data["is_not_training"] = p==0.0f;

    if(p==0.0f) return x;
    
    auto mask = (torch::rand({x.size(-2), x.size(-1)}, torch::TensorOptions().device(x.device()).dtype(torch::kHalf))>p).to(x.dtype()).div_(1-p);

    ctx->save_for_backward({mask});
    return x.mul(mask);
  }
  static torch::autograd::variable_list backward(
		torch::autograd::AutogradContext* ctx,
		const torch::autograd::variable_list& grad_output
  ) {
    if(ctx->saved_data["is_not_training"].toBool()) 
      return {grad_output[0], torch::Tensor()};
    
    return {grad_output[0].mul_(ctx->get_saved_variables()[0]), torch::Tensor()};
  }
};

torch::Tensor dropout(
  torch::Tensor& x,
  float p
) {
  return Dropout::apply(x, p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dropout", &dropout);
}