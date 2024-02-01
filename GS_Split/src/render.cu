#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include "config.h"
#include "auxiliary.h"


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


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	const int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	const bool* visibility_filter,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_filter[idx])
		return;

	// Find this Gaussian's offset in buffer for writing keys/values.
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	uint2 rect_min, rect_max;

	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	// For each tile that the bounding rect overlaps, emit a 
	// key/value pair. The key is |  tile ID  |      depth      |,
	// and the value is the ID of the Gaussian. Sorting the values 
	// with this key yields Gaussian IDs in a list, such that they
	// are first sorted by tile and then by depth. 
	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
RenderForwardCUDAKernel(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ means2D,
	const float* __restrict__ features,
	const float3* __restrict__ conics,
	const float* __restrict__ opacities,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_feature)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_conics[BLOCK_SIZE];
	__shared__ float collected_opacities[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float F[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = means2D[coll_id];
			collected_conics[block.thread_rank()] = conics[coll_id];
			collected_opacities[block.thread_rank()] = opacities[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float3 conic = collected_conics[j];
			float opacity = collected_opacities[j];
			float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opacity * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				F[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_feature[ch * H * W + pix_id] = F[ch];
	}
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
RenderBackwardCUDAKernel(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ means2D,
	const float3* __restrict__ conics,
	const float* __restrict__ opacities,
	const float* __restrict__ features,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dout_feature,
	float* __restrict__ dL_dfeatures,
	float2* __restrict__ dL_dmeans2D,
	float3* __restrict__ dL_dconics,
	float* __restrict__ dL_dopacities
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_conics[BLOCK_SIZE];
	__shared__ float collected_opacities[BLOCK_SIZE];
	__shared__ float collected_features[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dout_feature[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = means2D[coll_id];
			collected_conics[block.thread_rank()] = conics[coll_id];
			collected_opacities[block.thread_rank()] = opacities[coll_id];
			for (int i = 0; i < C; i++)
				collected_features[i * BLOCK_SIZE + block.thread_rank()] = features[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float3 conic = collected_conics[j];
			const float opacity = collected_opacities[j];
			const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opacity * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian features and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_features[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dfeatures[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Helpful reusable temporary variables
			const float dL_dG = opacity * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * conic.x - gdy * conic.y;
			const float dG_ddely = -gdy * conic.z - gdx * conic.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmeans2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmeans2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconics[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconics[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconics[global_id].z, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacities[global_id]), G * dL_dalpha);
		}
	}
}

int RenderForwardCUDA(
	const int P,
	const int width, const int height,
	const float* features,
	const float* depths,
	const int32_t* radii,
	const float2* means2D,
	const float3* conics,
	const float* opacities,
	const int32_t* tiles_touched,
	const bool* visibility_filter,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	float* out_feature,
	bool debug)
{
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	char* scanning_space;
	uint32_t* point_offsets;
    cudaMalloc((void**)&scanning_space, P * sizeof(char));
    cudaMalloc((void**)&point_offsets, P * sizeof(uint32_t));

	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, point_offsets, point_offsets, P);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(scanning_space, temp_storage_bytes, tiles_touched, point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		means2D,
		depths,
		point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		visibility_filter,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

    RenderForwardCUDAKernel<NUM_CHANNELS> <<<tile_grid, block >> > (
		imgState.ranges,
		binningState.point_list,
		width, height,
		means2D,
		features,
		conics,
		opacities,
		imgState.accum_alpha,
		imgState.n_contrib,
		out_feature);
	
	cudaFree(scanning_space);
	cudaFree(point_offsets);

	return num_rendered;
}


void RenderBackwardCUDA(
	const int P, const int num_rendered,
	const int width, const int height,
	const float* features,
	const float2* means2D,
	const float3* conics,
	const float* opacities,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dout_feature,
	float* dL_dfeatures,
	float2* dL_dmeans2D,
	float3* dL_dconics,
	float* dL_dopacities,
	bool debug)
{
	BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	RenderBackwardCUDAKernel<NUM_CHANNELS> <<<tile_grid, block>>>(
		imgState.ranges,
		binningState.point_list,
		width, height,
		means2D,
		conics,
		opacities,
		features,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dout_feature,
		dL_dfeatures,
		dL_dmeans2D,
		dL_dconics,
		dL_dopacities
	);
}