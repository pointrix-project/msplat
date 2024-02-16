/**
 * @file ewa_project.h
 * @brief Wrapper function of EWA projection.
 */
#pragma once

#include <torch/extension.h>

/**
 * @brief 
 * 
 * @param xyz 
 * @param cov3d 
 * @param viewmat 
 * @param camparam 
 * @param uv 
 * @param W 
 * @param H 
 * @param visibility_status 
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
 */
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

/**
 * @brief 
 * 
 * @param xyz 
 * @param cov3d 
 * @param viewmat 
 * @param camparam 
 * @param radius 
 * @param dL_dconic 
 * @return std::tuple<torch::Tensor, torch::Tensor> 
 */
std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& radius,
    const torch::Tensor& dL_dconic
);
