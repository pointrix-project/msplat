
#include <utils.h>
#include <compute_sh.h>
#include <compute_sh_kernel.h>

std::tuple<torch::Tensor, torch::Tensor> 
computeSHForward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status
){
    CHECK_INPUT(shs);
    CHECK_INPUT(view_dirs);
    CHECK_INPUT(visibility_status);

    const int P = shs.size(0);
    auto float_opts = shs.options().dtype(torch::kFloat32);
    auto bool_ops = shs.options().dtype(torch::kBool);
    torch::Tensor colors = torch::zeros({P, 3}, float_opts);
    torch::Tensor clamped = torch::ones({P, 3}, bool_ops);

    if(P != 0)
    {
        computeSHForwardCUDA(
            P, 
            shs.contiguous().data_ptr<float>(), 
            degree, 
            view_dirs.contiguous().data_ptr<float>(), 
            visibility_status.contiguous().data_ptr<bool>(), 
            colors.data_ptr<float>(), 
            clamped.data_ptr<bool>()
        );
    }
    
    return std::make_tuple(colors, clamped);
}

std::tuple<torch::Tensor, torch::Tensor> 
computeSHBackward(
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& view_dirs,
    const torch::Tensor& visibility_status,
    const torch::Tensor& clamped,
    const torch::Tensor& dL_dcolors
){
    CHECK_INPUT(shs);
    CHECK_INPUT(view_dirs);
    CHECK_INPUT(visibility_status);
    CHECK_INPUT(dL_dcolors);

    const int P = shs.size(0);
    const int S = shs.size(1);
    auto float_opts = shs.options().dtype(torch::kFloat32);
    torch::Tensor dL_dshs = torch::zeros({P, S, 3}, float_opts);
    torch::Tensor dL_dvdirs = torch::zeros({P, 3}, float_opts);

    if(P != 0)
    {
        computeSHBackwardCUDA(
            P, 
            shs.contiguous().data_ptr<float>(), 
            degree, 
            view_dirs.contiguous().data_ptr<float>(), 
            visibility_status.contiguous().data_ptr<bool>(),
            clamped.contiguous().data_ptr<bool>(),
            dL_dcolors.contiguous().data_ptr<float>(), 
            dL_dshs.data_ptr<float>(),
            dL_dvdirs.data_ptr<float>()
        );
    }
    
    return std::make_tuple(dL_dshs, dL_dvdirs);
}

