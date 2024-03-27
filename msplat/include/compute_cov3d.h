/**
 * @file compute_cov3d.h
 * @brief Wrapper function of 3D covariance matrices computation.
 *
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching the CUDA kernel to compute 3D covariance matrices in a
 * forward pass.
 *
 * @param[in] scales        3D scales for each point.
 * @param[in] uquats        3D rotations (unit quaternions) for each
 * point.
 * @param[in] visible       Visibility status of each point.
 * @return torch::Tensor <br> <b>cov3d</b> The upper-right corner of the 3D
 * covariance matrix, stored in a vector. <br>
 */
torch::Tensor computeCov3DForward(const torch::Tensor &scales,
                                  const torch::Tensor &uquats,
                                  const torch::Tensor &visible);

/**
 * @brief Launching the CUDA kernel to compute 3D covariance matrices in a
 * backward pass.
 *
 * @param[in] scales        3D scales for each point.
 * @param[in] uquats        3D rotations (unit quaternions) for each
 * point.
 * @param[in] visible       Visibility status of each point.
 * @param[in] dL_dcov3Ds    Gradients of the loss with respect to the 3D
 * covariance matrices.
 * @return std::tuple<torch::Tensor, torch::Tensor> <br> (1)<b>dL_dscales</b>
 * Gradients of the loss with respect to scales. <br> (2)<b>dL_duquats</b>
 * Gradients of the loss with respect to rotations. <br>
 */
std::tuple<torch::Tensor, torch::Tensor>
computeCov3DBackward(const torch::Tensor &scales,
                     const torch::Tensor &uquats,
                     const torch::Tensor &visible,
                     const torch::Tensor &dL_dcov3Ds);