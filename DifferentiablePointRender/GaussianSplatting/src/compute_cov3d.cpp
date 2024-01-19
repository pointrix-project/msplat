
/**
 * @file compute_cov3d.cpp
 * @author Jian Gao
 * @brief Wrapper function of 3D covariance matrices coputation.
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <utils.h>
#include <compute_cov3d.h>
#include <compute_cov3d_kernel.h>

torch::Tensor
computeCov3DForward(
  const torch::Tensor& scales,
  const torch::Tensor& uquats,
  const torch::Tensor& visibility_status
){

    CHECK_INPUT(scales);
    CHECK_INPUT(uquats);
    CHECK_INPUT(visibility_status);

    const int P = scales.size(0);
    auto float_opts = scales.options().dtype(torch::kFloat32);
    torch::Tensor cov3Ds = torch::zeros({P, 6}, float_opts);
    if(P != 0)
    {
        computeCov3DForwardCUDA(
            P,
            scales.contiguous().data_ptr<float>(),
            uquats.contiguous().data_ptr<float>(),
            visibility_status.contiguous().data_ptr<bool>(),
            cov3Ds.data_ptr<float>());
    }

    return cov3Ds;
}



std::tuple<torch::Tensor, torch::Tensor>
computeCov3DBackward(
    const torch::Tensor& scales,
	const torch::Tensor& uquats,
	const torch::Tensor& visibility_status,
	const torch::Tensor& dL_dcov3Ds
){

    CHECK_INPUT(scales);
    CHECK_INPUT(uquats);
    CHECK_INPUT(visibility_status);
    CHECK_INPUT(dL_dcov3Ds);

    const int P = scales.size(0);
    auto float_opts = scales.options().dtype(torch::kFloat32);
    torch::Tensor dL_dscales = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_duquats = torch::zeros({P, 4}, float_opts);
    
    if(P != 0)
    {
        computeCov3DBackwardCUDA(
                P,
                scales.contiguous().data_ptr<float>(),
                uquats.contiguous().data_ptr<float>(),
                visibility_status.contiguous().data_ptr<bool>(),
                dL_dcov3Ds.contiguous().data_ptr<float>(),
                dL_dscales.data_ptr<float>(),
                dL_duquats.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dscales, dL_duquats);
}
