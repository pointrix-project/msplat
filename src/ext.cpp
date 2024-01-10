/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "preprocess.h"
#include "render.h"
#include "sh.h"
#include "rasterizer_impl.h"
#include "config.h"
#include <cuda_runtime_api.h>

std::function<char*(size_t N)> resizeFunc(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessForward(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3Ds_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
  const int image_height,
  const int image_width,
	const bool debug){
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor depths = torch::zeros({P, 1}, float_opts);
  torch::Tensor radii = torch::zeros({P}, int_opts);
  torch::Tensor means2D = torch::zeros({P, 2}, float_opts);
  torch::Tensor cov3Ds = torch::zeros({P, 6}, float_opts);
  torch::Tensor conics = torch::zeros({P, 3}, float_opts);
  torch::Tensor tiles_touched = torch::zeros({P}, int_opts);
  
  if(P != 0)
  {
	  PreprocessForwardCUDA(
	    P,
      W, H,
      means3D.contiguous().data_ptr<float>(),
      scales.contiguous().data_ptr<float>(),
      rotations.contiguous().data_ptr<float>(),
      cov3Ds_precomp.contiguous().data_ptr<float>(), 
      viewmatrix.contiguous().data_ptr<float>(), 
      projmatrix.contiguous().data_ptr<float>(),
      tan_fovx,
      tan_fovy,
      depths.contiguous().data_ptr<float>(),
      radii.contiguous().data_ptr<int>(),
      (float2*)means2D.contiguous().data_ptr<float>(),
      cov3Ds.contiguous().data_ptr<float>(),
      (float3*)conics.contiguous().data_ptr<float>(),
      tiles_touched.contiguous().data_ptr<int>(),
      debug);
  }
  return std::make_tuple(depths, radii, means2D, cov3Ds, conics, tiles_touched);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessBackward(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3Ds,
	const torch::Tensor& cov3Ds_precomp,
	const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
  const int image_height,
  const int image_width,
  const torch::Tensor& dL_ddepths,
  const torch::Tensor& dL_dmeans2D,
  const torch::Tensor& dL_dcov3Ds,
  const torch::Tensor& dL_dconics,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  PreprocessBackwardCUDA(P, 
	  W, H, 
	  (const float3*)means3D.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  rotations.data_ptr<float>(),
	  cov3Ds.data_ptr<float>(),
	  cov3Ds_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  dL_ddepths.contiguous().data_ptr<float>(),
	  (const float2*)dL_dmeans2D.contiguous().data_ptr<float>(),
	  (const float3*)dL_dconics.contiguous().data_ptr<float>(),  
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  (float4*)dL_drotations.contiguous().data_ptr<float>(),
	  dL_dcov3Ds.contiguous().data_ptr<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dcov3Ds);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
RenderForward(
  const torch::Tensor& features,
  const torch::Tensor& depths,
  const torch::Tensor& radii,
  const torch::Tensor& means2D,
  const torch::Tensor& conics,
  const torch::Tensor& opacities,
  const torch::Tensor& tiles_touched,
  const torch::Tensor& visibility_filter,
  const int image_height,
  const int image_width,
	const bool debug)
{
  const int P = features.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = features.options().dtype(torch::kInt32);
  auto float_opts = features.options().dtype(torch::kFloat32);

  torch::Tensor out_feature = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> binningFunc = resizeFunc(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunc(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
    rendered = RenderForwardCUDA(
      P, W, H,
      features.contiguous().data_ptr<float>(), 
      depths.contiguous().data_ptr<float>(),
      radii.contiguous().data_ptr<int>(),
      (const float2*)means2D.contiguous().data_ptr<float>(),
      (const float3*)conics.contiguous().data_ptr<float>(),
      opacities.contiguous().data_ptr<float>(),
      tiles_touched.contiguous().data_ptr<int>(),
      visibility_filter.contiguous().data_ptr<bool>(),
      binningFunc,
      imgFunc,
      out_feature.contiguous().data_ptr<float>(),
      debug);
  }
  return std::make_tuple(rendered, out_feature, binningBuffer, imgBuffer);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderBackward(
  const torch::Tensor& features,
  const torch::Tensor& means2D,
  const torch::Tensor& conics,
  const torch::Tensor& opacities,
  const torch::Tensor& dL_dout_feature,
	const int num_rendered,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = features.size(0);
  const int H = dL_dout_feature.size(1);
  const int W = dL_dout_feature.size(2);
  
  auto float_opts = features.options().dtype(torch::kFloat32);
  torch::Tensor dL_dfeatures = torch::zeros({P, NUM_CHANNELS}, float_opts);
  torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, float_opts);
  torch::Tensor dL_dconics = torch::zeros({P, 3},float_opts);
  torch::Tensor dL_dopacities = torch::zeros({P, 1}, float_opts);
  
  if(P != 0)
  {  
	  RenderBackwardCUDA(
      P, num_rendered, W, H, 
      features.contiguous().data_ptr<float>(),
      (const float2*) means2D.contiguous().data_ptr<float>(),
      (const float3*) conics.contiguous().data_ptr<float>(),
      opacities.contiguous().data_ptr<float>(),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      dL_dout_feature.contiguous().data_ptr<float>(),
      dL_dfeatures.contiguous().data_ptr<float>(),
      (float2*)dL_dmeans2D.contiguous().data_ptr<float>(),  
      (float3*)dL_dconics.contiguous().data_ptr<float>(),
      dL_dopacities.contiguous().data_ptr<float>(),
      debug
    );
  }

  return std::make_tuple(dL_dfeatures, dL_dmeans2D, dL_dconics, dL_dopacities);
}



std::tuple<torch::Tensor, torch::Tensor>
ComputeColorFromSHForward(
  const int deg, 
  const torch::Tensor dirs, 
  const torch::Tensor shs,
  const torch::Tensor visibility_filter
){
  const int P = shs.size(0);
  auto float_opts = shs.options().dtype(torch::kFloat32);
  auto bool_ops = shs.options().dtype(torch::kBool);
  torch::Tensor colors = torch::zeros({P, 3}, float_opts);
  torch::Tensor clamped = torch::zeros({P, 3}, bool_ops);

  int M = 0;
  if(shs.size(0) != 0)
  {
    M = shs.size(1);
  }
  if (P != 0 && M != 0){
    ComputeColorFromSHForwardCUDA(
      P, deg, M,
      dirs.contiguous().data_ptr<float>(),
      shs.contiguous().data_ptr<float>(),
      visibility_filter.contiguous().data_ptr<bool>(),
      colors.contiguous().data_ptr<float>(),
      clamped.contiguous().data_ptr<bool>());
  }
  return std::make_tuple(colors, clamped);
}

std::tuple<torch::Tensor, torch::Tensor>
ComputeColorFromSHBackward(
  const int deg, 
  const torch::Tensor dirs, 
  const torch::Tensor shs,
  const torch::Tensor visibility_filter,
  const torch::Tensor clamped,
  torch::Tensor dL_dcolors
){
  const int P = shs.size(0);
  
  int M = 0;
  if(shs.size(0) != 0)
  {
    M = shs.size(1);
  }

  auto float_opts = shs.options().dtype(torch::kFloat32);
  auto bool_ops = shs.options().dtype(torch::kBool);

  torch::Tensor dL_ddirs = torch::zeros({P, 3}, float_opts);
  torch::Tensor dL_dshs = torch::zeros({P, M, 3}, float_opts);

  if (P != 0 && M != 0){
    ComputeColorFromSHBackwardCUDA(
      P, deg, M,
      dirs.contiguous().data_ptr<float>(),
      shs.contiguous().data_ptr<float>(),
      visibility_filter.contiguous().data_ptr<bool>(),
      clamped.contiguous().data_ptr<bool>(),
      dL_dcolors.contiguous().data_ptr<float>(),
      dL_ddirs.contiguous().data_ptr<float>(),
      dL_dshs.contiguous().data_ptr<float>()
    );
  }
  return std::make_tuple(dL_ddirs, dL_dshs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_forward", &PreprocessForward);
  m.def("preprocess_backward", &PreprocessBackward);
  m.def("render_forward", &RenderForward);
  m.def("render_backward", &RenderBackward);
  m.def("compute_sh_forward", &ComputeColorFromSHForward);
  m.def("compute_sh_backward", &ComputeColorFromSHBackward);
}