
#include <utils.h>
#include <ewa_project.h>
#include <ewa_project_kernel.h>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EWAProjectForward(
    const torch::Tensor& xyz,
    const torch::Tensor& cov3d,
    const torch::Tensor& viewmat,
    const torch::Tensor& camparam,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& xy,
    const int W, const int H,
    const torch::Tensor &visibility_status
){
    CHECK_INPUT(xyz);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(camparam);
    CHECK_INPUT(uv);
    CHECK_INPUT(depth);
    CHECK_INPUT(xy);
    CHECK_INPUT(visibility_status);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
     auto int_opts = xyz.options().dtype(torch::kInt32);

    torch::Tensor cov2d = torch::zeros({P, 3}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    torch::Tensor tiles_touched = torch::zeros({P}, int_opts);

    if(P != 0){
        EWAProjectForwardCUDA(
            P,
            xyz.contiguous().data_ptr<float>(),
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

// std::tuple<torch::Tensor, torch::Tensor>
// EWAProjectBackward(
//     const torch::Tensor& xyz,
//     const torch::Tensor& cov3d,
//     const torch::Tensor& viewmat,
//     const torch::Tensor& projmat,
//     const torch::Tensor& camparam,
//     const int W, const int H,
//     const torch::Tensor& visibility_status,
//     const torch::Tensor& dL_dcov2d,
//     const torch::Tensor& dL_dopacity
// ){
//     CHECK_INPUT(xyz);
//     CHECK_INPUT(cov3d);
//     CHECK_INPUT(viewmat);
//     CHECK_INPUT(projmat);
//     CHECK_INPUT(camparam);
//     CHECK_INPUT(visibility_status);
//     CHECK_INPUT(dL_dcov2d);
//     CHECK_INPUT(dL_dopacity);

//     const int P = xyz.size(0);
//     auto float_opts = xyz.options().dtype(torch::kFloat32);

//     torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);
//     torch::Tensor dL_dcov3d = torch::zeros({P, 3}, float_opts);

//     if(P != 0){
//         // Kernel
//     }

//     return std::make_tuple(dL_dxyz, dL_dcov3d);
// }