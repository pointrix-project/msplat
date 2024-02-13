
#include <utils.h>
#include <iostream>
#include <torch/torch.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void computeGaussianKeyCUDAKernel(
    const int P,
    const float2* uv,
    const float* depth,
    const int* radii,
    const int* cum_tiles_hit,
    const dim3 tile_grid,
    int64_t* isect_ids,
    int* gaussian_ids
){
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= P || radii[idx] <= 0)
        return;
    
    uint2 tile_min, tile_max;
    float2 center = uv[idx];
    get_rect(center, radii[idx], tile_min, tile_max, tile_grid);
    
    int cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    int64_t depth_id = (int64_t) * (int *)&(depth[idx]);
    for (int i = tile_min.y; i < tile_max.y; i++) {
        for (int j = tile_min.x; j < tile_max.x; j++) {
            int64_t tile_id = i * tile_grid.x + j;
            isect_ids[cur_idx] = (tile_id << 32) | depth_id;

            gaussian_ids[cur_idx] = idx;
            cur_idx++;
        }
    }
}


__global__ void computeTileGaussianRangeCUDAKernel(
    const int num_intersects, 
    const int64_t* isect_ids_sorted, 
    int2* tile_bins
){
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    
    int cur_tile_idx = (int)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }

    if (idx == 0)
        return;
    
    int prev_tile_idx = (int)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        
        return;
    }
}


std::tuple<torch::Tensor, torch::Tensor> 
computeGaussianKey(
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const int W, const int H, 
    const torch::Tensor& radii,
    const torch::Tensor& cum_tiles_hit
){ 
    // map_gaussian_to_intersects
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    const int P = uv.size(0);
    
    try
    {
        if(P == 0)
            throw std::runtime_error("The point number is 0.");
    } 
    catch(std::runtime_error &e)
    {
        printf("%s\n", e.what());
    }
   
    const int num_intersects = cum_tiles_hit[P-1].item<int>();
    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    auto int32_opts = uv.options().dtype(torch::kInt32);
    auto int64_opts = uv.options().dtype(torch::kInt64);
    torch::Tensor gaussian_ids = torch::zeros({num_intersects}, int32_opts);
    torch::Tensor isect_ids = torch::zeros({num_intersects}, int64_opts);

    computeGaussianKeyCUDAKernel<<<(P + 255) / 256, 256>>>(
        P,
        (float2*)uv.contiguous().data_ptr<float>(),
        depth.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        cum_tiles_hit.contiguous().data_ptr<int>(),
        tile_grid,
        isect_ids.data_ptr<int64_t>(),
        gaussian_ids.data_ptr<int>()
    );

    return std::make_tuple(isect_ids, gaussian_ids);
}

torch::Tensor
computeTileGaussianRange(
    const int W, const int H, 
    const torch::Tensor& cum_tiles_hit,
    const torch::Tensor& isect_ids_sorted
){
    // get_tile_bin_edges
    CHECK_INPUT(cum_tiles_hit);
    CHECK_INPUT(isect_ids_sorted);

    const int P = cum_tiles_hit.size(0);
    
    try
    {
        if(P == 0)
            throw std::runtime_error("The point number is 0.");
    } 
    catch(std::runtime_error &e)
    {
        printf("%s\n", e.what());
    }
   
    const int num_intersects = cum_tiles_hit[P-1].item<int>();
    int num_tiles = int((W + BLOCK_X - 1) / BLOCK_X) * int((H + BLOCK_Y - 1) / BLOCK_Y);

    auto int32_opts = cum_tiles_hit.options().dtype(torch::kInt32);
    torch::Tensor tile_bins = torch::zeros({num_tiles, 2}, int32_opts);

    computeTileGaussianRangeCUDAKernel<<<(num_intersects + 255)/256, 256>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2*)tile_bins.data_ptr<int>()
    );

    return tile_bins;
}
