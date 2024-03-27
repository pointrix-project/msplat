/**
 * @file ewa_project.h
 * @brief Wrapper function of EWA projection.
 */
#pragma once

#include <torch/extension.h>

/**
 * @brief Launching CUDA kernel to perform ewa projection in a forward pass.
 *
 * @param[in] xyz       The 3D coordinates of each point in the scene.
 * @param[in] cov3d     The upper-right corner of the 3D covariance matrices,
 * stored in a vector.
 * @param[in] intr      The camera intrinsic parameters [fx, fy, cx, cy].
 * @param[in] extr      The camera extrinsic parameters [R|T].
 * @param[in] uv        2D positions for each point in the image.
 * @param[in] W         The width of the image.
 * @param[in] H         The height of the image.
 * @param[in] visible   The visibility status of each point
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> (1)
 * <b>conic</b> The upper-right corner of the 2D covariance matrices, stored in
 * a vector. (2) <b>radius</b> Radius of the 2D planar Gaussian on the image.
 * (3) <b>tiles</b> Number of tiles covered by 2D planar Gaussians on the image.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(const torch::Tensor &xyz,
                  const torch::Tensor &cov3d,
                  const torch::Tensor &intr,
                  const torch::Tensor &extr,
                  const torch::Tensor &uv,
                  const int W,
                  const int H,
                  const torch::Tensor &visible);

/**
 * @brief Launching CUDA kernel to perform ewa projection in a backward pass.
 *
 * @param[in] xyz       3D position of the 3D Gaussians in the scene.
 * @param[in] cov3d     The upper-right corner of the 3D covariance matrices,
 * stored in a vector.
 * @param[in] intr      The camera intrinsic parameters [fx, fy, cx, cy].
 * @param[in] extr      The camera extrinsic parameters [R|T].
 * @param[in] radius    Radius of the 2D planar Gaussian on the image.
 * @param[in] dL_dconic Gradients of loss with respect to the conic.
 * @return std::tuple<torch::Tensor, torch::Tensor> (1) <b>dL_dxyz</b> Gradients
 * of loss with respect to xyz. (2) <b>dL_dcov3d</b> Gradients of loss with
 * respect to cov3d.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectBackward(const torch::Tensor &xyz,
                   const torch::Tensor &cov3d,
                   const torch::Tensor &intr,
                   const torch::Tensor &extr,
                   const torch::Tensor &radius,
                   const torch::Tensor &dL_dconic);
