
#include <utils.h>
#include <project_point.h>
#include <project_point_kernel.h>


std::tuple<torch::Tensor, torch::Tensor> 
projectPointsForward(
    const torch::Tensor& xyz,
    const torch::Tensor& viewmat,
    const torch::Tensor& projmat,
    const torch::Tensor& camparam,
    const int W, const int H,
    const float nearest,
    const float extent
){
    CHECK_INPUT(xyz);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(projmat);
    CHECK_INPUT(camparam);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor uv = torch::zeros({P, 2}, float_opts);
    torch::Tensor depth = torch::zeros({P, 1}, float_opts);

    if(P != 0)
    {
        projectPointForwardCUDA(
            P,
            xyz.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            projmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            W, H,
            (float2*)uv.data_ptr<float>(), 
            depth.data_ptr<float>(),
            nearest,
            extent
        );
    }

    return std::make_tuple(uv, depth);
}


torch::Tensor 
projectPointsBackward(
    const torch::Tensor& xyz,
    const torch::Tensor& viewmat,
    const torch::Tensor& projmat,
    const torch::Tensor& camparam,
    const int W, const int H,
    const torch::Tensor& uv,
    const torch::Tensor& depth,
    const torch::Tensor& dL_duv,
    const torch::Tensor& dL_ddepth 
){
    CHECK_INPUT(xyz);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(projmat);
    CHECK_INPUT(camparam);

    const int P = xyz.size(0);
    auto float_opts = xyz.options().dtype(torch::kFloat32);
    torch::Tensor dL_dxyz = torch::zeros({P, 3}, float_opts);

    if(P != 0)
    {
        projectPointBackwardCUDA(
            P,
            xyz.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(),
            projmat.contiguous().data_ptr<float>(),
            camparam.contiguous().data_ptr<float>(),
            W, H,
            (float2*)uv.contiguous().data_ptr<float>(), 
            depth.contiguous().data_ptr<float>(),
            (float2*)dL_duv.contiguous().data_ptr<float>(),
            dL_ddepth.contiguous().data_ptr<float>(),
            dL_dxyz.data_ptr<float>()
        );
    }

    return dL_dxyz;
}