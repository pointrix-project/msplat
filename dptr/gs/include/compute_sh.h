/**
 * @file compute_sh.h
 * @brief Wrapper function of computing RGB color from Spherical Harmonics(SHs).
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief 
 * 
 * @param shs 
 * @param degree 
 * @param view_dirs 
 * @param visibility_status 
 * @return std::tuple<torch::Tensor, torch::Tensor> 
 */
std::tuple<torch::Tensor, torch::Tensor> 
computeSHForward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status
);

/**
 * @brief 
 * 
 * @param shs 
 * @param degree 
 * @param view_dirs 
 * @param visibility_status 
 * @param clamped 
 * @param dL_dcolors 
 * @return std::tuple<torch::Tensor, torch::Tensor> 
 */
std::tuple<torch::Tensor, torch::Tensor> 
computeSHBackward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status,
    const torch::Tensor& clamped,
    const torch::Tensor& dL_dcolors
);
