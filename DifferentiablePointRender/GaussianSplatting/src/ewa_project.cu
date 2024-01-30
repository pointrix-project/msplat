
#include <utils.h>
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>

namespace cg = cooperative_groups;


__global__ void EWAProjectForwardCUDAKernel(
    int P,
    const dim3 grid,
    const float* cov3d,
    const float* viewmat,
    const float* camparam,
    const float* uv,
    const float* depth,
    const float* xy,
    const int W, const int H,
    const bool* visibility_status,
    float* cov2d,
    int* radii,
    int* tiles_touched
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visibility_status[idx])
        return;

    float focal_x = camparam[0];
    float focal_y = camparam[1];

    // EWA project
    float3 t = {uv[2*idx], uv[2*idx+1], depth[idx]};
    t.x = t.x * t.z;
    t.y = t.y * t.z;

    glm::mat3 Jmat = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 Wmat = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]);

    glm::mat3 Tmat = Wmat * Jmat;
    glm::mat3 Vrk = glm::mat3(
        cov3d[6 * idx + 0], cov3d[6 * idx + 1], cov3d[6 * idx + 2],
        cov3d[6 * idx + 1], cov3d[6 * idx + 3], cov3d[6 * idx + 4],
        cov3d[6 * idx + 2], cov3d[6 * idx + 4], cov3d[6 * idx + 5]);

    glm::mat3 ewa_cov = glm::transpose(Tmat) * glm::transpose(Vrk) * Tmat;

    float3 cov = { float(ewa_cov[0][0]) + 0.3f, float(ewa_cov[0][1]), float(ewa_cov[1][1]) + 0.3f};
    
    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;

    // Compute extent in screen space
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    uint2 rect_min, rect_max;
    float2 mean2D = { xy[2 * idx], xy[2 * idx + 1] };
    getRect(mean2D, my_radius, rect_min, rect_max, grid);

    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // cov2d 
    float det_inv = 1.f / det;
    cov2d[3 * idx] = cov.z * det_inv;
    cov2d[3 * idx + 1] = -cov.y * det_inv;
    cov2d[3 * idx + 2] = cov.x * det_inv;

    // radius
    radii[idx] = my_radius;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


__global__ void EWAProjectBackCUDAKernel(
    int P,
    const float* cov3d,
    const float* viewmat,
    const float* camparam,
    const float* uv,
    const float* depth,
    const float* xy,
    const int W, const int H,
    const bool* visibility_status,
    const float* dL_dcov2d,
    float* dL_dcov3d,
    float* dL_dd
){
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visibility_status[idx])
        return;

    // dL_dcov2d [N, 3]
    // dL_dcov3d [N, 6]
    float fx = camparam[0];
    float fy = camparam[1];

    // dL_dcov3d = dL_dcov2d * dcov2d_dcov3d
    float dcov2d_0_dcov3d_0 = fx * fx * viewmat[0] * viewmat[0] / (depth[idx] * depth[idx]); // fx**2*v0**2/d**2
    float dcov2d_0_dcov3d_1 = 2 * fx * fx * viewmat[0] * viewmat[1] / (depth[idx] * depth[idx]); // 2*fx**2*v0*v1/d**2
    float dcov2d_0_dcov3d_2 = 2 * fx * fx * viewmat[0] * viewmat[2] / (depth[idx] * depth[idx]); //2*fx**2*v0*v2/d**2
    float dcov2d_0_dcov3d_3 = fx * fx * viewmat[1] * viewmat[1] / (depth[idx] * depth[idx]); // fx**2*v1**2/d**2
    float dcov2d_0_dcov3d_4 = 2 * fx * fx * viewmat[1] * viewmat[2] / (depth[idx] * depth[idx]); // 2*fx**2*v1*v2/d**2
    float dcov2d_0_dcov3d_5 = fx * fx * viewmat[2] * viewmat[2] / (depth[idx] * depth[idx]); // fx**2*v2**2/d**2

    float dcov2d_1_dcov3d_0 = fx * fy * viewmat[0] * viewmat[4] / (depth[idx] * depth[idx]); //fx*fy*v0*v4/d**2
    float dcov2d_1_dcov3d_1 = fx * fy * viewmat[0] * viewmat[5] / (depth[idx] * depth[idx]) 
                            + fx * fy * viewmat[1] * viewmat[4] / (depth[idx] * depth[idx]);//fx*fy*v0*v5/d**2 + fx*fy*v1*v4/d**2
    float dcov2d_1_dcov3d_2 = fx * fy * viewmat[0] * viewmat[6] / (depth[idx] * depth[idx]) 
                            + fx * fy * viewmat[2] * viewmat[4] / (depth[idx] * depth[idx]);//fx*fy*v0*v6/d**2 + fx*fy*v2*v4/d**2
    float dcov2d_1_dcov3d_3 = fx * fy * viewmat[1] * viewmat[5] / (depth[idx] * depth[idx]);//fx*fy*v1*v5/d**2
    float dcov2d_1_dcov3d_4 = fx * fy * viewmat[1] * viewmat[6] / (depth[idx] * depth[idx])
                            + fx * fy * viewmat[2] * viewmat[5] / (depth[idx] * depth[idx]);//fx*fy*v1*v6/d**2 + fx*fy*v2*v5/d**2
    float dcov2d_1_dcov3d_5 = fx * fy * viewmat[2] * viewmat[6] / (depth[idx] * depth[idx]);//fx*fy*v2*v6/d**2
    
    float dcov2d_2_dcov3d_0 = fy * fy * viewmat[4] * viewmat[4] / (depth[idx] * depth[idx]);//fy**2*v4**2/d**2
    float dcov2d_2_dcov3d_1 = 2 * fy * fy * viewmat[4] * viewmat[5] / (depth[idx] * depth[idx]);//2*fy**2*v4*v5/d**2
    float dcov2d_2_dcov3d_2 = 2 * fy * fy * viewmat[4] * viewmat[6] / (depth[idx] * depth[idx]);//2*fy**2*v4*v6/d**2
    float dcov2d_2_dcov3d_3 = fy * fy * viewmat[5] * viewmat[5] / (depth[idx] * depth[idx]);//fy**2*v5**2/d**2
    float dcov2d_2_dcov3d_4 = 2 * fy * fy * viewmat[5] * viewmat[6] / (depth[idx] * depth[idx]);//2*fy**2*v5*v6/d**2
    float dcov2d_2_dcov3d_5 = fy * fy * viewmat[6] * viewmat[6] / (depth[idx] * depth[idx]);//fy**2*v6**2/d**2

    dL_dcov3d[6 * idx + 0] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_0
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_0
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_0;
    dL_dcov3d[6 * idx + 1] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_1
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_1
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_1;
    dL_dcov3d[6 * idx + 2] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_2
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_2
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_2;
    dL_dcov3d[6 * idx + 3] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_3
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_3
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_3;
    dL_dcov3d[6 * idx + 4] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_4
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_4
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_4;
    dL_dcov3d[6 * idx + 5] = dL_dcov2d[3 * idx + 0] * dcov2d_0_dcov3d_5
                           + dL_dcov2d[3 * idx + 1] * dcov2d_1_dcov3d_5
                           + dL_dcov2d[3 * idx + 2] * dcov2d_2_dcov3d_5;
    
    // dL_dd = dL_dcov2d * dcov2d_dd
    
    // ===============================Dcov2d/Dd===============================
    // dcov2d_0_dd = -2*fx**2*(v0*(cov3d_0*v0 + cov3d_1*v1 + cov3d_2*v2) 
    //             + v1*(cov3d_1*v0 + cov3d_3*v1 + cov3d_4*v2) 
    //             + v2*(cov3d_2*v0 + cov3d_4*v1 + cov3d_5*v2))/d**3
    // dcov2d_1_dd = -2*fx*fy*(v4*(cov3d_0*v0 + cov3d_1*v1 + cov3d_2*v2) 
    //             + v5*(cov3d_1*v0 + cov3d_3*v1 + cov3d_4*v2) 
    //             + v6*(cov3d_2*v0 + cov3d_4*v1 + cov3d_5*v2))/d**3
    // dcov2d_2_dd = -2*fy**2*(v4*(cov3d_0*v4 + cov3d_1*v5 + cov3d_2*v6) 
    //             + v5*(cov3d_1*v4 + cov3d_3*v5 + cov3d_4*v6) 
    //             + v6*(cov3d_2*v4 + cov3d_4*v5 + cov3d_5*v6))/d**3
    
    float dcov2d_0_dd = -2*fx*fx*(
                      viewmat[0]*(cov3d[0]*viewmat[0] + cov3d[1]*viewmat[1] + cov3d[2]*viewmat[2]) 
                      + viewmat[1]*(cov3d[1]*viewmat[0] + cov3d[3]*viewmat[1] + cov3d[4]*viewmat[2]) 
                      + viewmat[2]*(cov3d[2]*viewmat[0] + cov3d[4]*viewmat[1] + cov3d[5]*viewmat[2]))
                      /(depth[idx] * depth[idx] * depth[idx]);
    float dcov2d_1_dd = -2*fx*fy*(
                      viewmat[4]*(cov3d[0]*viewmat[0] + cov3d[1]*viewmat[1] + cov3d[2]*viewmat[2]) 
                      + viewmat[5]*(cov3d[1]*viewmat[0] + cov3d[3]*viewmat[1] + cov3d[4]*viewmat[2]) 
                      + viewmat[6]*(cov3d[2]*viewmat[0] + cov3d[4]*viewmat[1] + cov3d[5]*viewmat[2]))
                      /(depth[idx] * depth[idx] * depth[idx]);
    float dcov2d_2_dd = -2*fy*fy*(
                      viewmat[4]*(cov3d[0]*viewmat[4] + cov3d[1]*viewmat[5] + cov3d[2]*viewmat[6]) 
                      + viewmat[5]*(cov3d[1]*viewmat[4] + cov3d[3]*viewmat[5] + cov3d[4]*viewmat[6]) 
                      + viewmat[6]*(cov3d[2]*viewmat[4] + cov3d[4]*viewmat[5] + cov3d[5]*viewmat[6]))
                      /(depth[idx] * depth[idx] * depth[idx]);
    
    dL_dd[idx] = dL_dcov2d[3*idx + 0] * dcov2d_0_dd 
               + dL_dcov2d[3*idx + 1] * dcov2d_1_dd 
               + dL_dcov2d[3*idx + 1] * dcov2d_2_dd;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor &visibility_status
){
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(xy);
    CHECK_INPUT(visibility_status);

    const int P = uv.size(0);
    auto float_opts = uv.options().dtype(torch::kFloat32);
     auto int_opts = uv.options().dtype(torch::kInt32);

    torch::Tensor cov2d = torch::zeros({P, 3}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);

    if(P != 0){
        const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);

        EWAProjectForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
            P, 
            tile_grid,
            cov3d.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            xy.contiguous().data_ptr<float>(),
            W, H,
            visibility_status.contiguous().data_ptr<bool>(),
            cov2d.data_ptr<float>(),
            radii.data_ptr<int>(),
            tiles_touched.data_ptr<int>()
        );
    }

    return std::make_tuple(cov2d, radii, tiles_touched);
}

std::tuple<torch::Tensor, torch::Tensor>
EWAProjectBackward(
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor& visibility_status,
    const torch::Tensor& dL_dcov2d
){
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(xy);
    CHECK_INPUT(visibility_status);
    CHECK_INPUT(dL_dcov2d);

    const int P = uv.size(0);
    auto float_opts = uv.options().dtype(torch::kFloat32);
    torch::Tensor dL_dcov3d = torch::zeros({P, 6}, float_opts);
    torch::Tensor dL_dd = torch::zeros({P, 1}, float_opts);

    if(P != 0){
        // Kernel
        EWAProjectBackCUDAKernel<<<(P + 255) / 256, 256 >>>(
            P, 
            cov3d.contiguous().data_ptr<float>(), 
            viewmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            uv.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            xy.contiguous().data_ptr<float>(),
            W, H,
            visibility_status.contiguous().data_ptr<bool>(),
            dL_dcov2d.contiguous().data_ptr<float>(),
            dL_dcov3d.data_ptr<float>(),
            dL_dd.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dcov3d, dL_dd);
}