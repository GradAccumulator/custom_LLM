#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;

class Tanh : public torch::autograd::Function<Tanh> {
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor& x
  ) {
    auto exp = x.exp();
    auto recip = exp.reciprocal();
    auto numer = exp.sub(recip);
    auto denom = exp.add(recip);
    ctx->save_for_backward({numer, denom});
    return numer/denom;
  }
private:
  #define NUMER 0
  #define DENOM 1
public:
  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output
  ) {
    const auto& saved = ctx->get_saved_variables();
    return { grad_output[0].mul_(1-(saved[NUMER].pow_(2)).div_(saved[DENOM].pow_(2))) };
  }
};

torch::Tensor tanh(torch::Tensor& x) {
  return Tanh::apply(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tanh", &tanh);
}