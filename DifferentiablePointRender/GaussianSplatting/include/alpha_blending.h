
#pragma once

#include <torch/extension.h>


torch::Tensor
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

