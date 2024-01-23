

#pragma once

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor &visibility_status
);

std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& projmat,
    const torch::Tensor& camparam,
    const int W, const int H,
    const torch::Tensor& visibility_status,
    const torch::Tensor& dL_dcov2d,
    const torch::Tensor& dL_dopacity
);
