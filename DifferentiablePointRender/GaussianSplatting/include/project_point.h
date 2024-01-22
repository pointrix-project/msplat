/**
 * @file project_point.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CUDA_PROJECT_POINTS_H_INCLUDED
#define CUDA_PROJECT_POINTS_H_INCLUDED

#include <torch/extension.h>


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

# endif