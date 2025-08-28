#include <torch/torch.h>
#include <iostream>
using namespace torch::indexing;
using namespace std;

namespace mine {
	template<typename T>
	void print(T value, bool print_size = true) {
		if (typeid(T) == typeid(torch::Tensor) && print_size) {
			cout << value.sizes() << endl;
			return;
		}
		cout << value << endl;
	}
}

void main() {
	/*dSoftmaxi_dScoresi = dSoftmaxi_dScoresi
			- (weight_i.reshape({ batch, num_heads, seq_len, 1, seq_len })
				* weight_i.reshape({ batch, num_heads, seq_len, seq_len, 1 }))
			[~eye];*/
	int batch = 10;
	int num_heads = 2;
	int seq_len = 7;
	auto eye = torch::eye(seq_len).reshape({ 1,1,1,seq_len,seq_len });
	auto weight_i = torch::randn({ batch, num_heads, seq_len ,seq_len});
}