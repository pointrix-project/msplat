/**
 * @file compute_sh.h
 * @brief Wrapper function of computing RGB color from Spherical Harmonics(SHs).
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching the CUDA kernel to compute RGB color from SHs in a forward pass.
 * 
 * @param shs                     Spherical Harmonics(SHs).
 * @param degree                  The degree of SHs.
 * @param view_dirs               Normalized view direction.
 * @param visible                The visibility status of each point.
 * @return std::tuple<torch::Tensor, torch::Tensor>                    <br>
 *         (1) <b>colors</b>      The view-dependent RGB color.        <br>
 *         (2) <b>clamped</b>     The color is clamped or not.         <br>
 */
std::tuple<torch::Tensor, torch::Tensor> 
computeSHForward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visible
);

/**
 * @brief Launching the CUDA kernel to compute RGB color from SHs in a backward pass.
 * 
 * @param shs                     Spherical Harmonics(SHs).
 * @param degree                  The degree of SHs.
 * @param view_dirs               Normalized view direction.
 * @param visible                The visibility status of each point
 * @param clamped                 The RGB color is clamped or not.
 * @param dL_dcolors              Gradients of the loss with respect to RGB colors.
 * @return std::tuple<torch::Tensor, torch::Tensor>                                 <br>
 *         (1) <b>dL_dshs</b>     Gradients of the loss with respect to SHs.
 *         (2) <b>dL_dvdirs</b>   Gradients of the loss with respect to view direction.
 */
std::tuple<torch::Tensor, torch::Tensor> 
computeSHBackward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visible,
    const torch::Tensor& clamped,
    const torch::Tensor& dL_dcolors
);
