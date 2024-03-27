
/**
 * @file sort_gaussian.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <torch/torch.h>
#include <utils.h>


namespace cg = cooperative_groups;

__global__ void computeGaussianKeyCUDAKernel(const int P,
                                             const float2 *uv,
                                             const float *depth,
                                             const int *radius,
                                             const int *tiles,
                                             const dim3 tile_grid,
                                             int64_t *gaussian_key,
                                             int *gaussian_idx) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= P || radius[idx] <= 0)
        return;

    uint2 tile_min, tile_max;
    float2 center = uv[idx];
    get_rect(center, radius[idx], tile_min, tile_max, tile_grid);

    int cur_idx = (idx == 0) ? 0 : tiles[idx - 1];
    int64_t depth_id = (int64_t) * (int *)&(depth[idx]);
    for (int i = tile_min.y; i < tile_max.y; i++) {
        for (int j = tile_min.x; j < tile_max.x; j++) {
            int64_t tile_id = i * tile_grid.x + j;
            gaussian_key[cur_idx] = (tile_id << 32) | depth_id;
            gaussian_idx[cur_idx] = idx;
            cur_idx++;
        }
    }
}

__global__ void
computeTileGaussianRangeCUDAKernel(const int num_intersects,
                                   const int64_t *isect_idx_sorted,
                                   int2 *tile_range) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;

    int cur_tile_idx = (int)(isect_idx_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_range[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_range[cur_tile_idx].y = num_intersects;
    }

    if (idx == 0)
        return;

    int prev_tile_idx = (int)(isect_idx_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_range[prev_tile_idx].y = idx;
        tile_range[cur_tile_idx].x = idx;

        return;
    }
}

std::tuple<torch::Tensor, torch::Tensor>
computeGaussianKey(const torch::Tensor &uv,
                   const torch::Tensor &depth,
                   const int W,
                   const int H,
                   const torch::Tensor &radius,
                   const torch::Tensor &tiles) {
    // map_gaussian_to_intersects
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(radius);
    CHECK_INPUT(tiles);

    const int P = uv.size(0);

    try {
        if (P == 0)
            throw std::runtime_error("The point number is 0.");
    } catch (std::runtime_error &e) {
        printf("%s\n", e.what());
    }

    const int num_intersects = tiles[P - 1].item<int>();
    const dim3 tile_grid(
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    auto int32_opts = uv.options().dtype(torch::kInt32);
    auto int64_opts = uv.options().dtype(torch::kInt64);
    torch::Tensor gaussian_idx = torch::zeros({num_intersects}, int32_opts);
    torch::Tensor gaussian_key = torch::zeros({num_intersects}, int64_opts);

    computeGaussianKeyCUDAKernel<<<(P + 255) / 256, 256>>>(
        P,
        (float2 *)uv.contiguous().data_ptr<float>(),
        depth.contiguous().data_ptr<float>(),
        radius.contiguous().data_ptr<int>(),
        tiles.contiguous().data_ptr<int>(),
        tile_grid,
        gaussian_key.data_ptr<int64_t>(),
        gaussian_idx.data_ptr<int>());

    return std::make_tuple(gaussian_key, gaussian_idx);
}

torch::Tensor computeTileGaussianRange(const int W,
                                       const int H,
                                       const torch::Tensor &tiles,
                                       const torch::Tensor &key_sorted) {
    // get_tile_bin_edges
    CHECK_INPUT(tiles);
    CHECK_INPUT(key_sorted);

    const int P = tiles.size(0);

    try {
        if (P == 0)
            throw std::runtime_error("The point number is 0.");
    } catch (std::runtime_error &e) {
        printf("%s\n", e.what());
    }

    const int num_intersects = tiles[P - 1].item<int>();
    int num_tiles =
        int((W + BLOCK_X - 1) / BLOCK_X) * int((H + BLOCK_Y - 1) / BLOCK_Y);

    auto int32_opts = tiles.options().dtype(torch::kInt32);
    torch::Tensor tile_range = torch::zeros({num_tiles, 2}, int32_opts);

    computeTileGaussianRangeCUDAKernel<<<(num_intersects + 255) / 256, 256>>>(
        num_intersects,
        key_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_range.data_ptr<int>());

    return tile_range;
}
