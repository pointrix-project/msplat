
#pragma once

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
alphaBlendingForward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& gaussian_idx_sorted,
    const torch::Tensor& tile_bins,
    const float bg,
    const int W, const int H
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& gaussian_idx_sorted,
    const torch::Tensor& tile_bins,
    const float bg,
    const int W, const int H,
    const torch::Tensor &final_T,
    const torch::Tensor &ncontrib,
    const torch::Tensor& dL_drendered
);
