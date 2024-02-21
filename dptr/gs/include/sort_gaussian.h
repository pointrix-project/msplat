/**
 * @file sort_gaussian.h
 * @brief Sort 2D planar Gaussians base on [tile|depth].
 */

#pragma once

#include <torch/extension.h>

/**
 * @brief Launching CUDA kernel to compute key for Gaussians based on the tile
 * index and depth.
 *
 * @param[in] uv                  2D position of each point after projection.
 * @param[in] depth               The depth value of each point.
 * @param[in] W                   The width of the image.
 * @param[in] H                   The height of the image.
 * @param[in] radius              Radius of the 2D planar Gaussian on the image.
 * @param[in] tiles               Number of tiles covered by 2D planar Gaussians
 * on the image.
 * @return std::tuple<torch::Tensor, torch::Tensor>  <br> (1)<b>gaussian_key</b>
 * Key of Gaussian. <br> (2)<b>gaussian_idx</b> The gaussian index. If the same
 * gaussian is in more than one tile, it will be repeated multiple times.
 */
std::tuple<torch::Tensor, torch::Tensor>
computeGaussianKey(const torch::Tensor &uv,
                   const torch::Tensor &depth,
                   const int W,
                   const int H,
                   const torch::Tensor &radius,
                   const torch::Tensor &tiles);

/**
 * @brief
 *
 * @param[in] W                     The width of the image.
 * @param[in] H                     The height of the image.
 * @param[in] tiles                 Number of tiles covered by 2D planar
 * Gaussians on the image.
 * @param[in] key_sorted            Soted key of Gaussians.
 * @return torch::Tensor <br> <b>tile_range</b>  Ranges of indices in idx_sorted
 * for Gaussians in each tile.
 */
torch::Tensor computeTileGaussianRange(const int W,
                                       const int H,
                                       const torch::Tensor &tiles,
                                       const torch::Tensor &key_sorted);
