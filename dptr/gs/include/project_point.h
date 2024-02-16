/**
 * @file project_point.h
 * @brief Wrapper function of point projection.
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching CUDA kernel to perform point projection in a forward pass.
 * 
 * @param xyz         3D position of the 3D Gaussians in the scene.
 * @param viewmat     The world to camera view transformation matrix.
 * @param projmat     The world to screen transform matrix. 
 * @param camparam    The intrinsic parameters of the camera [fx, fy, cx, cy].
 * @param W           The width of the image.
 * @param H           The height of the image.
 * @param nearest     Nearest threshold for frustum culling.
 * @param extent      Extent threshold for frustum culling.
 * @return std::tuple<torch::Tensor, torch::Tensor>                               <br>
 *         (1) <b>uv</b> The 2D pixel coordinates of each point after projection. <br>
 *         (2) <b>depth</b> The depth value of each point.                        <br>
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
 * @brief Launching CUDA kernel to perform point projection in a forward pass.
 * 
 * @param[in] xyz         3D position of the 3D Gaussians in the scene.
 * @param[in] viewmat     The world to camera view transformation matrix.
 * @param[in] projmat     The world to screen transform matrix. 
 * @param[in] camparam    The intrinsic parameters of the camera [fx, fy, cx, cy].
 * @param[in] W           The width of the image.
 * @param[in] H           The height of the image.
 * @param[in] nearest     Nearest threshold for frustum culling.
 * @param[in] extent      Extent threshold for frustum culling.
 * @return std::tuple<torch::Tensor, torch::Tensor>                               <br>
 *         (1) <b>uv</b>    2D position of each point after projection.           <br>
 *         (2) <b>depth</b> The depth value of each point.                        <br>
 */
/**
 * @brief Launching CUDA kernel to perform point projection in a backward pass.
 * 
 * @param[in] xyz         3D position of the 3D Gaussians in the scene.
 * @param[in] viewmat     The world to camera view transformation matrix.
 * @param[in] projmat     The world to screen transform matrix. 
 * @param[in] camparam    The intrinsic parameters of the camera [fx, fy, cx, cy].
 * @param[in] W           The width of the image.
 * @param[in] H           The height of the image.
 * @param[in] uv          2D position of each point after projection.
 * @param[in] depth       The depth value of each point.
 * @param[in] dL_duv      Gradients of loss with respect to uv.
 * @param[in] dL_ddepth   Gradients of loss with respect to depth.
 * @return torch::Tensor 
*          <b>dL_dxyz</b> Gradients of loss with respect to xyz.
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