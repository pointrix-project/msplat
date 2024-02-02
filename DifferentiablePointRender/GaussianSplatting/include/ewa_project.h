
#pragma once

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const int W, const int H,
    const torch::Tensor &visibility_status
);

std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& radii,
    const torch::Tensor& dL_dconic
);
