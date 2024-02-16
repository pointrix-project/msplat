/**
 * @file compute_cov3d.h
 * @brief Wrapper function of 3D covariance matrices coputation.
 * 
 */

#pragma once

#include <torch/extension.h>


/**
 * @brief Wrapper function for launching the CUDA kernel to compute 3D covariance matrices in a forward pass.
 *
 * @param[in] scales              Tensor containing 3D scales for each point.
 * @param[in] uquats              Tensor containing 3D rotations (unit quaternions) for each point.
 * @param[in] visibility_status   Tensor indicating the visibility status of each point.
 * @return torch::Tensor          Output Tensor for storing the computed 3D covariance matrices.
 */
torch::Tensor
computeCov3DForward(
    const torch::Tensor& scales,
    const torch::Tensor& uquats,
    const torch::Tensor& visibility_status
);

/**
 * @brief Wrapper function for launching the CUDA kernel to compute gradients in a backward pass
 *        of 3D covariance computation.
 *
 * @param[in] scales              Tensor of 3D scales for each point.
 * @param[in] uquats              Tensor of 3D rotations (unit quaternions) for each point.
 * @param[in] visibility_status   Tensor indicating the visibility status of each point.
 * @param[in] dL_dcov3Ds          Gradients of the loss with respect to the 3D covariance matrices.
 * @return std::tuple<torch::Tensor, torch::Tensor> Output Tensor for storing the gradients of the loss with respect to scales and rotations
 */
std::tuple<torch::Tensor, torch::Tensor> 
computeCov3DBackward(
    const torch::Tensor& scales,
    const torch::Tensor& uquats,
    const torch::Tensor& visibility_status,
    const torch::Tensor& dL_dcov3Ds
);