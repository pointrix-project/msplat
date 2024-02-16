/**
 * @file alpha_blending.h
 * @brief Wrapper function of Alpha Blending for sorted 2D planar Gaussian in a tile based manner
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Wrapper function for launching the CUDA kernel to perform alpha blending in a forward pass.
 * 
 * @param[in] uv                    Tensor containing 2D positions for each point in the image.
 * @param[in] conic                 Tensor containing inverse 2D covariances for each point in the image.
 * @param[in] opacity               Tensor containing opacity values for each point.
 * @param[in] feature               Tensor containing features for each point to be alpha blended.
 * @param[in] idx_sorted            Tensor containing indices of Gaussian points sorted according to [tile_id|depth].
 * @param[in] tile_bins             Tensor specifying the range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
 * @param[in] bg                    Background color for alpha blending.
 * @param[in] W                     Width of the image.
 * @param[in] H                     Height of the image.
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
 *                                  A tuple containing three tensors: rendered features, final_T, ncontrib.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
alphaBlendingForward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& idx_sorted,
    const torch::Tensor& tile_bins,
    const float bg,
    const int W, const int H
);

/**
 * @brief Wrapper function for launching the CUDA kernel to compute gradients in a backward pass
 *        of alpha blending.
 * 
 * @param[in] uv                    Tensor containing 2D positions for each point in the image.
 * @param[in] conic                 Tensor containing inverse 2D covariances for each point in the image.
 * @param[in] opacity               Tensor containing opacity values for each point.
 * @param[in] feature               Tensor containing features for each point.
 * @param[in] idx_sorted            Tensor containing indices of Gaussian points sorted according to [tile_id|depth].
 * @param[in] tile_bins             Tensor specifying the range of indices in idx_sorted for Gaussians participating in alpha blending in each tile.
 * @param[in] bg                    Background color.
 * @param[in] W                     Width of the image.
 * @param[in] H                     Height of the image.
 * @param[in] final_T               Tensor containing the final transparency for each pixel.
 * @param[in] ncontrib              Tensor containing the number of contributions per point.
 * @param[in] dL_drendered          Tensor containing the gradient of the loss with respect to the rendered image.
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
 *                                  A tuple containing four tensors: gradients with respect to uv, conic, opacity, and feature.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& idx_sorted,
    const torch::Tensor& tile_bins,
    const float bg,
    const int W, const int H,
    const torch::Tensor &final_T,
    const torch::Tensor &ncontrib,
    const torch::Tensor& dL_drendered
);
