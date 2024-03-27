/**
 * @file project_point.h
 * @brief Wrapper function of point projection.
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching CUDA kernel to perform point projection in a forward pass.
 *
 * @param[in] xyz         3D position of the 3D Gaussians in the scene.
 * @param[in] intr        The camera intrinsic parameters [fx, fy, cx, cy].
 * @param[in] extr        The camera extrinsic parameters [R|T].
 * @param[in] W           The width of the image.
 * @param[in] H           The height of the image.
 * @param[in] nearest     Nearest threshold for frustum culling.
 * @param[in] extent      Extent threshold for frustum culling.
 * @return std::tuple<torch::Tensor, torch::Tensor> <br> (1) <b>uv</b> The 2D
 * pixel coordinates of each point after projection. <br> (2) <b>depth</b> The
 * depth value of each point. <br>
 */
std::tuple<torch::Tensor, torch::Tensor>
projectPointsForward(const torch::Tensor &xyz,
                     const torch::Tensor &intr,
                     const torch::Tensor &extr,
                     const int W,
                     const int H,
                     const float nearest,
                     const float extent);

/**
 * @brief Launching CUDA kernel to perform point projection in a backward pass.
 *
 * @param[in] xyz         3D position of the 3D Gaussians in the scene.
 * @param[in] intr        The camera intrinsic parameters [fx, fy, cx, cy].
 * @param[in] extr        The camera extrinsic parameters [R|T].
 * @param[in] W           The width of the image.
 * @param[in] H           The height of the image.
 * @param[in] uv          2D position of each point after projection.
 * @param[in] depth       The depth value of each point.
 * @param[in] dL_duv      Gradients of loss with respect to uv.
 * @param[in] dL_ddepth   Gradients of loss with respect to depth.
 * @return torch::Tensor <br> (1)<b>dL_dxyz</b> Gradients of loss with respect
 * to xyz. (2)<b>dL_dintr</b> Gradients of loss with respect to intrinsic
 * parameters. (3)<b>dL_dextr</b> Gradients of loss with respect to extrinsic
 * parameters.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectPointsBackward(const torch::Tensor &xyz,
                      const torch::Tensor &intrinsic,
                      const torch::Tensor &extrinsic,
                      const int W,
                      const int H,
                      const torch::Tensor &uv,
                      const torch::Tensor &depth,
                      const torch::Tensor &dL_duv,
                      const torch::Tensor &dL_ddepth);