

#ifndef CUDA_COMPUTE_SH_H_INCLUDED
#define CUDA_COMPUTE_SH_H_INCLUDED

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> 
computeSHForward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status
);

std::tuple<torch::Tensor, torch::Tensor> 
computeSHBackward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status,
    const torch::Tensor& clamped,
    const torch::Tensor& dL_dcolors
);

#endif