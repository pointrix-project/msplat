
#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

namespace cg = cooperative_groups;


template<uint32_t CNum>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
alphaBlendingForwardCUDAKernel(
    const int P,
    const float2* __restrict__ uv,
    const float3* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ feature,
    const int* __restrict__ gaussian_idx_sorted,
    const int2* __restrict__ tile_bins,
    const float bg, const int C,
    const int W, const int H,
    const dim3 tile_grid, 
    float* __restrict__ final_T,
    int* __restrict__ ncontrib,
    float* __restrict__ rendered_feature
)
{
    auto block = cg::this_thread_block();
    int32_t tile_id = block.group_index().y * tile_grid.x + block.group_index().x;
    uint2 pix = { 
        block.group_index().x * BLOCK_X + block.thread_index().x, 
        block.group_index().y * BLOCK_Y + block.thread_index().y
    };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
    const int c_num = min(CNum, C);

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    int2 range = tile_bins[tile_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_uv[BLOCK_SIZE];
    __shared__ float3 collected_conic[BLOCK_SIZE];
    __shared__ float collected_opacity[BLOCK_SIZE];
    
    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float F[CNum] = {0};

    for(int i = 0; i < rounds; i++, toDo-= BLOCK_SIZE)
    {
        int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

        // Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = gaussian_idx_sorted[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_uv[block.thread_rank()] = uv[coll_id];
			collected_conic[block.thread_rank()] = conic[coll_id];
			collected_opacity[block.thread_rank()] = opacity[coll_id];
		}
		block.sync();

        for(int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            contributor ++;
            float2 vec = { 
                collected_uv[j].x - pixf.x, 
                collected_uv[j].y - pixf.y
                };
            float power = -0.5f * (collected_conic[j].x * vec.x * vec.x + collected_conic[j].z * vec.y * vec.y) - collected_conic[j].y * vec.x * vec.y;
            float alpha = min(0.99f, collected_opacity[j] * exp(power));

            if(power > 0 || alpha < 1.0 / 255.0f)
                continue;

            float next_T = T * (1 - alpha);
            if(next_T < 0.0001f)
            {
                done = true;
                continue;
            }

            for(int k = 0; k < c_num; k++)
                F[k] += feature[k * P + collected_id[j]] * alpha * T;
            
            T = next_T;

            last_contributor = contributor;
        }
    }

    if(inside)
    {
        final_T[pix_id] = T;
        ncontrib[pix_id] = last_contributor;
        for(int k = 0; k < c_num; k++)
            rendered_feature[k * H * W + pix_id] = F[k] + T * bg;
    }
}

torch::Tensor
alphaBlendingForward(
    const torch::Tensor& uv,
    const torch::Tensor& conic,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& gaussian_idx_sorted,
    const torch::Tensor& tile_bins,
    const float bg,
    const int W, const int H
){
    CHECK_INPUT(uv);
    CHECK_INPUT(conic);
    CHECK_INPUT(opacity);
    CHECK_INPUT(feature);
    CHECK_INPUT(gaussian_idx_sorted);
    CHECK_INPUT(tile_bins);

    const int P = feature.size(0);
    const int C = feature.size(1);

    auto int_opts = feature.options().dtype(torch::kInt32);
    auto float_opts = feature.options().dtype(torch::kFloat32);
    torch::Tensor rendered_feature = torch::zeros({C, H, W}, float_opts);
    torch::Tensor final_T = torch::zeros({H, W}, float_opts);
    torch::Tensor ncontrib = torch::zeros({H, W}, int_opts);

    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // [C, N]
    torch::Tensor feature_permute = feature.transpose(0, 1);

    int C0 = 0;
    while(C0 < C){
        size_t feature_data_offset = C0 * P;
        size_t render_data_offset = C0 * H * W;

        if(C - C0 <= 3){
            alphaBlendingForwardCUDAKernel<3> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 3;
        }
        else if(C - C0 <= 6){
            alphaBlendingForwardCUDAKernel<6> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 6;
        }
        else if(C - C0 <= 12){
            alphaBlendingForwardCUDAKernel<12> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 12;
        }
        else if(C - C0 <= 18){
            alphaBlendingForwardCUDAKernel<18> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 18;
        }
        else if(C - C0 <= 24){
            alphaBlendingForwardCUDAKernel<24> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 24;
        }
        else{
            alphaBlendingForwardCUDAKernel<32> <<<tile_grid, block>>>(
                P,
                (float2*)uv.contiguous().data_ptr<float>(),
                (float3*)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + feature_data_offset,
                gaussian_idx_sorted.contiguous().data_ptr<int>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                bg, C - C0, W, H, tile_grid,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + render_data_offset
            );
            C0 += 32;
        }
    }

    return rendered_feature;
}