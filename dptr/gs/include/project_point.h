/**
 * @file project_point.h
 * @brief 
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief 
 * 
 * @param xyz 
 * @param viewmat 
 * @param projmat 
 * @param camparam 
 * @param W 
 * @param H 
 * @param nearest 
 * @param extent 
 * @return std::tuple<torch::Tensor, torch::Tensor> 
 */
std::tuple<torch::Tensor, torch::Tensor> 
projectPointsForward(
    const torch::Tensor& xyz,
    const torch::Tensor& viewmat,
    const torch::Tensor& projmat,
    const torch::Tensor& camparam,
    const int W, const int H,
    const float nearest,
    const float extent
);

/**
 * @brief 
 * 
 * @param xyz 
 * @param viewmat 
 * @param projmat 
 * @param camparam 
 * @param W 
 * @param H 
 * @param uv 
 * @param depth 
 * @param dL_duv 
 * @param dL_ddepth 
 * @return torch::Tensor 
 */
torch::Tensor 
projectPointsBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& viewmat,
    const torch::Tensor& projmat,
    const torch::Tensor& camparam,
    const int W, const int H,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& dL_duv,
    const torch::Tensor& dL_ddepth 
);