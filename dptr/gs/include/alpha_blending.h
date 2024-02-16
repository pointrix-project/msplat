/**
 * @file alpha_blending.h
 * @brief Wrapper function of Alpha Blending for sorted 2D planar Gaussian in a tile based manner
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching the CUDA kernel to perform alpha blending in a forward pass.
 * 
 * @param[in] uv                    2D positions for each point in the image.
 * @param[in] conic                 Inverse 2D covariances for each point in the image.
 * @param[in] opacity               Opacity for each point.
 * @param[in] feature               Features for each point to be alpha blended.
 * @param[in] idx_sorted            Indices of Gaussian points sorted according to [tile_id|depth].
 * @param[in] tile_range             Ranges of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
 * @param[in] bg                    Background color.
 * @param[in] W                     Width of the image.
 * @param[in] H                     Height of the image.
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>                             <br>
 *         (1) <b>feature map</b>   Rendered feature maps.                                     <br>
 *         (2) <b>final_T</b>       Final transparency of each pixel.                          <br>
 *         (3) <b>ncontrib</b>      Number of gausses involved in alpha blending on each tile  <br>
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
alphaBlendingForward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& idx_sorted,
    const torch::Tensor& tile_range,
    const float bg,
    const int W, const int H
);

/**
 * @brief Launching the CUDA kernel to perform alpha blending in a backward pass.
 * 
 * @param[in] uv                    2D positions for each point in the image.
 * @param[in] conic                 Inverse 2D covariances for each point in the image.
 * @param[in] opacity               Opacity for each point.
 * @param[in] feature               Features for each point.
 * @param[in] idx_sorted            Indices of Gaussian points sorted according to [tile_id|depth].
 * @param[in] tile_range             Range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
 * @param[in] bg                    Background color.
 * @param[in] W                     Width of the image.
 * @param[in] H                     Height of the image.
 * @param[in] final_T               Final transparency of each pixel.
 * @param[in] ncontrib              Number of contributions per point.
 * @param[in] dL_drendered          Gradient of the loss with respect to the rendered image.
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> <br>
 *         (1)<b>dL_duv</b>      gradients with respect to uv.                <br>
 *         (2)<b>dL_dconic</b>   gradients with respect to conic.             <br>
 *         (3)<b>dL_dopacity</b> gradients with respect to opacity.           <br>
 *         (4)<b>dL_dfeature</b> gradients with respect to feature.           <br>
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& idx_sorted,
    const torch::Tensor& tile_range,
    const float bg,
    const int W, const int H,
    const torch::Tensor &final_T,
    const torch::Tensor &ncontrib,
    const torch::Tensor& dL_drendered
);
