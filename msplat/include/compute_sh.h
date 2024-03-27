/**
 * @file compute_sh.h
 * @brief Wrapper function of computing RGB color from Spherical Harmonics(SHs).
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching the CUDA kernel to compute RGB color from SHs in a forward
 * pass.
 *
 * @param shs               Spherical Harmonics(SHs).
 * @param view_dirs         Normalized view direction.
 * @param visible           The visibility status of each point.
 * @return torch::Tensor <b>value</b> The result of SH evaluation.
 */
torch::Tensor
computeSHForward(const torch::Tensor &shs,
                 const torch::Tensor &view_dirs,
                 const torch::Tensor &visible);

/**
 * @brief Launching the CUDA kernel to compute RGB color from SHs in a backward
 * pass.
 *
 * @param shs           Spherical Harmonics(SHs).
 * @param view_dirs     Normalized view direction.
 * @param visible       The visibility status of each point
 * @param dL_dvalue     Gradients of the loss with respect to evaluted value.
 * colors.
 * @return std::tuple<torch::Tensor, torch::Tensor> <br> (1) <b>dL_dshs</b>
 * Gradients of the loss with respect to SHs. (2) <b>dL_dvdirs</b> Gradients
 * of the loss with respect to view direction.
 */
std::tuple<torch::Tensor, torch::Tensor>
computeSHBackward(const torch::Tensor &shs,
                  const torch::Tensor &view_dirs,
                  const torch::Tensor &visible,
                  const torch::Tensor &dL_dcolors);
