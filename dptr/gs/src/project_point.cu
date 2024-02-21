/**
 * @file project_point_kernel.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <utils.h>


namespace cg = cooperative_groups;

/**
 * @brief CUDA kernel for projecting points onto the screen in a forward pass.
 *
 * @param[in] P             Number of points
 * @param[in] xyz           Coordinates of the points
 * @param[in] viewmat       Camera's view matrix, transforms points from world
 * coordinates to camera coordinates
 * @param[in] projmat       Camera's projection matrix, transforms points from
 * world coordinates to normalized device coordinates (NDC)
 * @param[in] camparam      Camera parameters, [fx, fy, cx, cy]
 * @param[in] W             Width of the image
 * @param[in] H             Height of the image
 * @param[out] uv           Coordinates of the points on the image
 * @param[out] depths       Depths of the points
 * @param[in] nearest       Culling points with depth less than this value
 * @param[in] extent        Culling points outside the extent
 */
__global__ void projectPointForwardCUDAKernel(const int P,
                                              const float *xyz,
                                              const float *viewmat,
                                              const float *projmat,
                                              const float *camparam,
                                              const int W,
                                              const int H,
                                              float2 *uv,
                                              float *depths,
                                              const float nearest,
                                              const float extent) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 pt = {xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]};

    float4 p_hom = transform_point_4x4(projmat, pt);
    float p_w = 1.0f / (p_hom.w + 1e-7);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    float3 p_view = transform_point_4x3(viewmat, pt);

    bool near_culling = p_view.z <= nearest;
    bool extent_culling = false;
    if (extent > 0)
        extent_culling = p_proj.x < -extent || p_proj.x > extent ||
                         p_proj.y < -extent || p_proj.y > extent;

    if (near_culling || extent_culling)
        return;

    uv[idx] = {ndc_to_pixel(p_proj.x, W, camparam[2]),
               ndc_to_pixel(p_proj.y, H, camparam[3])};
    depths[idx] = p_view.z;
}

__global__ void projectPointBackwardCUDAKernel(const int P,
                                               const float *xyz,
                                               const float *viewmat,
                                               const float *projmat,
                                               const float *camparam,
                                               const int W,
                                               const int H,
                                               const float2 *uv,
                                               const float *depth,
                                               const float2 *dL_duv,
                                               const float *dL_ddepth,
                                               float3 *dL_dxyz) {
    auto idx = cg::this_grid().thread_rank();

    // depth == 0 means culled
    if (idx >= P || depth[idx] == 0)
        return;

    float3 dL_dxyz_idx;
    dL_dxyz_idx.x = dL_ddepth[idx] * viewmat[2];
    dL_dxyz_idx.y = dL_ddepth[idx] * viewmat[6];
    dL_dxyz_idx.z = dL_ddepth[idx] * viewmat[10];

    // dpw_dx
    float3 pt = {xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]};
    float4 p_hom = transform_point_4x4(projmat, pt);
    float pw = 1.0f / (p_hom.w + 1e-7);

    float2 dL_dndc = {0.5 * W * dL_duv[idx].x, 0.5 * H * dL_duv[idx].y};

    float3 dpw_dxyz;
    dpw_dxyz.x = -pw * pw * projmat[3];
    dpw_dxyz.y = -pw * pw * projmat[7];
    dpw_dxyz.z = -pw * pw * projmat[11];

    dL_dxyz_idx.x += dL_dndc.x * (projmat[0] * pw + p_hom.x * dpw_dxyz.x);
    dL_dxyz_idx.y += dL_dndc.x * (projmat[4] * pw + p_hom.x * dpw_dxyz.y);
    dL_dxyz_idx.z += dL_dndc.x * (projmat[8] * pw + p_hom.x * dpw_dxyz.z);

    dL_dxyz_idx.x += dL_dndc.y * (projmat[1] * pw + p_hom.y * dpw_dxyz.x);
    dL_dxyz_idx.y += dL_dndc.y * (projmat[5] * pw + p_hom.y * dpw_dxyz.y);
    dL_dxyz_idx.z += dL_dndc.y * (projmat[9] * pw + p_hom.y * dpw_dxyz.z);

    dL_dxyz[idx] = dL_dxyz_idx;
}

std::tuple<torch::Tensor, torch::Tensor>
projectPointsForward(const torch::Tensor &xyz,
                     const torch::Tensor &viewmat,
                     const torch::Tensor &projmat,
                     const torch::Tensor &camparam,
                     const int W,
                     const int H,
                     const float nearest,
                     const float extent) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(projmat);
    CHECK_INPUT(camparam);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor uv = torch::zeros({P, 2}, float_opts);
    torch::Tensor depth = torch::zeros({P, 1}, float_opts);

    if (P != 0) {
        projectPointForwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            xyz.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            projmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            W,
            H,
            (float2 *)uv.data_ptr<float>(),
            depth.data_ptr<float>(),
            nearest,
            extent);
    }

    return std::make_tuple(uv, depth);
}

torch::Tensor projectPointsBackward(const torch::Tensor &xyz,
                                    const torch::Tensor &viewmat,
                                    const torch::Tensor &projmat,
                                    const torch::Tensor &camparam,
                                    const int W,
                                    const int H,
                                    const torch::Tensor &uv,
                                    const torch::Tensor &depth,
                                    const torch::Tensor &dL_duv,
                                    const torch::Tensor &dL_ddepth) {
    CHECK_INPUT(xyz);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(projmat);
    CHECK_INPUT(camparam);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);

    if (P != 0) {
        projectPointBackwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            xyz.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            projmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            W,
            H,
            (float2 *)uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            (float2 *)dL_duv.contiguous().data_ptr<float>(),
            dL_ddepth.contiguous().data_ptr<float>(),
            (float3 *)dL_dxyz.data_ptr<float>());
    }

    return dL_dxyz;
}