
/**
 * @file sort_gaussian.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <torch/torch.h>
#include <utils.h>
#include <cub/cub.cuh>


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

    int num_intersects = 0;
    if(P > 0)
        num_intersects = tiles[P - 1].item<int>();
    
    const dim3 tile_grid(
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

    auto int32_opts = uv.options().dtype(torch::kInt32);
    auto int64_opts = uv.options().dtype(torch::kInt64);
    torch::Tensor gaussian_idx = torch::zeros({num_intersects}, int32_opts);
    torch::Tensor gaussian_key = torch::zeros({num_intersects}, int64_opts);

    if(P > 0)
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

    int num_intersects = 0;
    if(P > 0)
        num_intersects = tiles[P - 1].item<int>();

    int num_tiles =
        int((W + BLOCK_X - 1) / BLOCK_X) * int((H + BLOCK_Y - 1) / BLOCK_Y);

    auto int32_opts = tiles.options().dtype(torch::kInt32);
    torch::Tensor tile_range = torch::zeros({num_tiles, 2}, int32_opts);

    if(num_intersects > 0)
        computeTileGaussianRangeCUDAKernel<<<(num_intersects + 255) / 256, 256>>>(
            num_intersects,
            key_sorted.contiguous().data_ptr<int64_t>(),
            (int2 *)tile_range.data_ptr<int>());

    return tile_range;
}

std::tuple<torch::Tensor, torch::Tensor>
sortGaussian(const torch::Tensor &uv,
             const torch::Tensor &depth,
             const int W,
             const int H,
             const torch::Tensor &radius,
             const torch::Tensor &tiles){
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(radius);
    CHECK_INPUT(tiles); 

    const int P = uv.size(0);

    // sum
    int* presum_tiles = nullptr;
    cudaMalloc(&presum_tiles, P * sizeof(int));

    void* d_temp_storage1 = nullptr;
    size_t temp_storage_bytes1 = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage1, 
        temp_storage_bytes1, 
        tiles.contiguous().data_ptr<int>(), 
        presum_tiles, 
        P);
    cudaMalloc(&d_temp_storage1, temp_storage_bytes1);

    cub::DeviceScan::InclusiveSum(
        d_temp_storage1, 
        temp_storage_bytes1, 
        tiles.contiguous().data_ptr<int>(), 
        presum_tiles, 
        P);

    cudaFree(d_temp_storage1);
    
    int num_intersects = 0;
    cudaMemcpy(&num_intersects, presum_tiles + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

    // calculate keys
    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    int num_tiles = tile_grid.x * tile_grid.y;

    auto int32_opts = uv.options().dtype(torch::kInt32);
    torch::Tensor sorted_idx = torch::zeros({num_intersects},int32_opts);
    torch::Tensor tile_range = torch::zeros({num_tiles, 2}, int32_opts);

    if(num_intersects > 0){

        int* gaussian_idx = nullptr;
        int64_t* gaussian_key = nullptr;
        int64_t* gaussian_key_sort = nullptr;
        cudaMalloc(&gaussian_idx, num_intersects * sizeof(int));
        cudaMalloc(&gaussian_key, num_intersects * sizeof(int64_t));
        cudaMalloc(&gaussian_key_sort, num_intersects * sizeof(int64_t));

        // key
        computeGaussianKeyCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float2 *)uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            radius.contiguous().data_ptr<int>(),
            presum_tiles,
            tile_grid,
            gaussian_key,
            gaussian_idx);
        
        // sort
        void* d_temp_storage2 = nullptr;
        size_t temp_storage_bytes2 = 0;
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage2, 
            temp_storage_bytes2, 
            gaussian_key, 
            gaussian_key_sort, 
            gaussian_idx, 
            sorted_idx.data_ptr<int>(), 
            num_intersects);
        cudaMalloc(&d_temp_storage2, temp_storage_bytes2);
        
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage2, 
            temp_storage_bytes2, 
            gaussian_key, 
            gaussian_key_sort, 
            gaussian_idx, 
            sorted_idx.data_ptr<int>(), 
            num_intersects);
        
        cudaFree(d_temp_storage2);

        // calculate tile_range
        computeTileGaussianRangeCUDAKernel<<<(num_intersects + 255) / 256, 256>>>(
            num_intersects,
            gaussian_key_sort,
            (int2 *)tile_range.data_ptr<int>());
        
        cudaFree(gaussian_idx);
        cudaFree(gaussian_key);
        cudaFree(gaussian_key_sort);
    }

    cudaFree(presum_tiles);
    
    return std::make_tuple(sorted_idx, tile_range);
}


// for memory alignment
char* allocAlignmentByteMemory(torch::Tensor& t, size_t &N){
    t.resize_({(long long)N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
}

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T> 
size_t required(size_t P)
{
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}

// memory alignment
struct titleInfo{
    int* tiles;
    int* presum_tiles;
    char* temp_storage;
    size_t temp_storage_bytes;

    static titleInfo fromChunk(char*& chunk, size_t P){
        titleInfo info;
        obtain(chunk, info.tiles, P, 128);
        obtain(chunk, info.presum_tiles, P, 128);
        cub::DeviceScan::InclusiveSum(nullptr, info.temp_storage_bytes, info.tiles, info.tiles, P);
        obtain(chunk, info.temp_storage, info.temp_storage_bytes, 128);
        
        return info;
    }
};

struct sortInfo{
    int* gaussian_idx;
    int* gaussian_idx_sort;
    int64_t* gaussian_key;
    int64_t* gaussian_key_sort;
    char* temp_storage;
    size_t temp_storage_bytes;

    static sortInfo fromChunk(char*& chunk, size_t P){
        sortInfo info;
        obtain(chunk, info.gaussian_key, P, 128);
        obtain(chunk, info.gaussian_key_sort, P, 128);
        obtain(chunk, info.gaussian_idx, P, 128);
        obtain(chunk, info.gaussian_idx_sort, P, 128);
        cub::DeviceRadixSort::SortPairs(
            nullptr, info.temp_storage_bytes,
            info.gaussian_key, info.gaussian_key_sort,
            info.gaussian_idx, info.gaussian_idx_sort, P);
        obtain(chunk, info.temp_storage, info.temp_storage_bytes, 128);

        return info;
    }
};

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

std::tuple<torch::Tensor, torch::Tensor>
sortGaussianFast(const torch::Tensor &uv,
                 const torch::Tensor &depth,
                 const int W,
                 const int H,
                 const torch::Tensor &radius,
                 const torch::Tensor &tiles){
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(radius);
    CHECK_INPUT(tiles); 

    const int P = uv.size(0);

    // Storing variables in structures and aligning them, to makes memory access faster
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor title_info_tensor = torch::empty({0}, options.device(device));
    torch::Tensor sort_info_tensor = torch::empty({0}, options.device(device));

    size_t chunk_size = required<titleInfo>(P);
    char* chunkptr = allocAlignmentByteMemory(title_info_tensor, chunk_size);
    titleInfo tile_info = titleInfo::fromChunk(chunkptr, P);

    // sum
    cudaMemcpy(tile_info.tiles, tiles.contiguous().data_ptr<int>(), P * sizeof(int), cudaMemcpyDeviceToDevice);
    cub::DeviceScan::InclusiveSum(tile_info.temp_storage, tile_info.temp_storage_bytes, tile_info.tiles, tile_info.presum_tiles, P);

    int num_intersects = 0;
    cudaMemcpy(&num_intersects, tile_info.presum_tiles + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    int num_tiles = tile_grid.x * tile_grid.y;

    // result tensor
    auto int32_opts = uv.options().dtype(torch::kInt32);
    torch::Tensor sorted_idx = torch::zeros({num_intersects},int32_opts);
    torch::Tensor tile_range = torch::zeros({num_tiles, 2}, int32_opts);

    if(num_intersects > 0){
        // calculate keys
        chunk_size = required<sortInfo>(num_intersects);
        chunkptr = allocAlignmentByteMemory(sort_info_tensor, chunk_size);
        sortInfo sort_info = sortInfo::fromChunk(chunkptr, num_intersects);
        
        // key
        computeGaussianKeyCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            (float2 *)uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            radius.contiguous().data_ptr<int>(),
            tile_info.presum_tiles,
            tile_grid,
            sort_info.gaussian_key,
            sort_info.gaussian_idx);
        
        // sort
        int bit = getHigherMsb(tile_grid.x * tile_grid.y);
        cub::DeviceRadixSort::SortPairs(
            sort_info.temp_storage, 
            sort_info.temp_storage_bytes, 
            sort_info.gaussian_key, 
            sort_info.gaussian_key_sort, 
            sort_info.gaussian_idx, 
            sort_info.gaussian_idx_sort, 
            num_intersects, 0, 32 + bit);

        // calculate tile_range
        computeTileGaussianRangeCUDAKernel<<<(num_intersects + 255) / 256, 256>>>(
            num_intersects,
            sort_info.gaussian_key_sort,
            (int2 *)tile_range.data_ptr<int>());
        
        cudaMemcpy(sorted_idx.data_ptr<int>(), sort_info.gaussian_idx_sort, num_intersects * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    return std::make_tuple(sorted_idx, tile_range);
}
