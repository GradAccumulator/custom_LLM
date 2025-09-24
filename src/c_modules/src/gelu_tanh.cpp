#include <torch/torch.h>
#include <torch/extension.h>
using namespace std;


/*



폐기



*/

class GELUTanh : public torch::autograd::Function<GELUTanh> {
private:
  static torch::Tensor tanh(
    torch::Tensor& x
  ) {
    auto exp = x.exp();
    auto recip = exp.reciprocal();
    
    return exp.sub(recip).div(exp.add(recip));
  }
  static torch::Tensor tanh_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& tanh_out
  ) {
    return grad_out.mul_(1-tanh_out.pow(2));
  }
private:
  #define SQRT__2_OVER_PI 0.79788
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor& x,
    double sqrt_2_over_pi
  ) {
    /*
    tanh = (1+self.tanh(self.sqrt_2_over_pi*(x+0.044715*(x**3))))
      return 0.5*x*tanh
    */
    auto& out = tanh(x.pow(3).mul_(0.044715).add_(x).mul_(SQRT__2_OVER_PI)).add_(1);


  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output
  ) {

  }
};