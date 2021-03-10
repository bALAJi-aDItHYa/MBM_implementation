#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <cuda_runtime_api.h>


std::vector<torch::Tensor> conv_forward(torch::Tensor input,
									  torch::Tensor weights,
									  int64_t kW, int64_t kH,
									  int64_t dW, int64_t dH,
									  int64_t padW, int64_t padH) {

	int64_t batch_size = input.size(0);
	int64_t nInputPlane = input.size(1);
	int64_t InputHeight = input.size(2);
	int64_t InputWidth = input.size(3);

	int64_t nOutputPlane = weights.size(0);
    int64_t outputHeight = (InputHeight + 2*padH - kH) / dH + 1;
    int64_t outputWidth = (InputWidth + 2*padW - kW) / dW + 1;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, nOutputPlane, outputHeight, outputWidth})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({nInputPlane*kW*kH, outputHeight*outputWidth})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({1, outputHeight*outputWidth})).cuda();

    //Reshaping weights --> outplanes * (inplanes * kH * kW)
    weights = weights.reshape(torch::IntArrayRef({nOutputPlane, nInputPlane*kW*kH})).cuda();

    for (int elt=0; elt<batch_size;elt++){
    	torch::Tensor input_n = input[elt];
    	columns = torch::im2col(input_n.clone(), torch::IntArrayRef({kW, kH}),
    											 torch::IntArrayRef({1,1}),
    											 torch::IntArrayRef({padW, padH}),
    											 torch::IntArrayRef({dW,dH})).cuda();

    	output[elt].add_(weights.mm(columns).reshape(torch::IntArrayRef({nOutputPlane, outputHeight, outputWidth})));
    }
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("c_forward", &conv_forward,"conv forward (CUDA)");
}