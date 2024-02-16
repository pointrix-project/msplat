/**
 * @file sort_gaussian.h
 * @brief 
 */

#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> 
computeGaussianKey(
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const int W, const int H, 
    const torch::Tensor& radius,
    const torch::Tensor& cum_tiles_hit
);

torch::Tensor
computeTileGaussianRange(
    const int W, const int H, 
    const torch::Tensor& cum_tiles_hit,
    const torch::Tensor& isect_idx_sorted
);
