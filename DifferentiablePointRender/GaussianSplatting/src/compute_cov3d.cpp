
#include<utils.h>
#include <compute_cov3d.h>
#include <compute_cov3d_kernel.h>

torch::Tensor
computeCov3DForward(
  const torch::Tensor& scales,
  const torch::Tensor& uquats,
  const torch::Tensor& visibility_staus
){

  CHECK_INPUT(scales);
  CHECK_INPUT(uquats);
  CHECK_INPUT(visibility_staus);

  const int P = scales.size(0);
  auto float_opts = scales.options().dtype(torch::kFloat32);
  torch::Tensor cov3Ds = torch::zeros({P, 6}, float_opts);
  if(P != 0)
  
  {
      computeCov3DForwardCUDA(
        P,
        scales.data_ptr<float>(),
        uquats.data_ptr<float>(),
        visibility_staus.data_ptr<bool>(),
        cov3Ds.data_ptr<float>());
  }

  return cov3Ds;
}

std::tuple<torch::Tensor, torch::Tensor>
computeCov3DBackward(
  const torch::Tensor& scales,
	const torch::Tensor& uquats,
	const torch::Tensor& visibility_staus,
	const torch::Tensor& dL_dcov3Ds
){

  CHECK_INPUT(scales);
  CHECK_INPUT(uquats);
  CHECK_INPUT(visibility_staus);
  CHECK_INPUT(dL_dcov3Ds);

  const int P = scales.size(0);
  auto float_opts = scales.options().dtype(torch::kFloat32);
  torch::Tensor dL_dscales = torch::zeros({P, 3}, float_opts);
  torch::Tensor dL_drotations = torch::zeros({P, 4}, float_opts);
  
  if(P != 0)
  {
      computeCov3DBackwardCUDA(
            P,
            scales.data_ptr<float>(),
            uquats.data_ptr<float>(),
            visibility_staus.data_ptr<bool>(),
            dL_dcov3Ds.data_ptr<float>(),
            dL_dscales.data_ptr<float>(),
            dL_drotations.data_ptr<float>()
    );
  }

  return std::make_tuple(dL_dscales, dL_drotations);
}
