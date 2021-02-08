// #include<torch/extension.h>
// #include<vector>
// #include<ATen/NativeFunctions.h>
// #include <ATen/Config.h>

// at::Tensor convolution_backward_weight(
// 	const at::Tensor &input,
// 	c10::ArrayRef<int64_t> weight_size,
//     const at::Tensor& grad_output,
//     c10::ArrayRef<int64_t> stride,
//     c10::ArrayRef<int64_t> padding,
//     c10::ArrayRef<int64_t> dilation,
//     int64_t groups,
//     bool benchmark,
//     bool deterministic,
//     bool allow_tf32) {

// 	return at::cudnn_convolution_backward_weight(
// 		weight_size,
// 		grad_output,
// 		input,
// 		padding,
// 		stride,
// 		dilation,
// 		groups,
// 		benchmark,
// 		deterministic,
// 		allow_tf32);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
// 	m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
// }

#include <torch/extension.h>
#include <c10/util/ArrayRef.h>

#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

at::Tensor convolution_backward_weight(
std::vector<int64_t> weight_size,
const at::Tensor& grad_output,
const at::Tensor& input,
std::vector<int64_t> padding,
std::vector<int64_t> stride,
std::vector<int64_t> dilation,
int64_t groups,
bool benchmark,
bool deterministic,
bool allow_tf32) {

return at::cudnn_convolution_backward_weight(
weight_size,
grad_output,
input,
padding,
stride,
dilation,
groups,
benchmark,
deterministic,
allow_tf32);
}

at::Tensor convolution_backward_input(
    std::vector<int64_t> input_size,
	const at::Tensor& grad_output,    
    const at::Tensor& weight,
    std::vector<int64_t> padding,
	std::vector<int64_t> stride,
	std::vector<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    return at::cudnn_convolution_backward_input(
        input_size,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("convolution_backward_weight", &convolution_backward_weight,"convolution backward weight");
m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}