/**
 * @file compute_cov3d_kernel.h
 * @author Jian Gao
 * @brief  CUDA kernel to compute 3D covariance matrices.
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CUDA_COMPUTE_COV3D_KERNEL_H_INCLUDED
#define CUDA_COMPUTE_COV3D_KERNEL_H_INCLUDED

/**
 * @brief Wrapper function for 3D covariance matrices computation in a forward pass.
 *
 * @param[in] P                    Number of points to process.
 * @param[in] scales               Array of 3D scales for each point.
 * @param[in] uquats               Array of 3D rotations (unit quaternions) for each point.
 * @param[in] visibility_status    Array indicating the visibility status of each point.
 * @param[out] cov3Ds              Output array for storing the computed 3D covariance matrices.
 */
void computeCov3DForwardCUDA(
	const int P, 
	const float* scales,
	const float* uquats,
	const bool* visibility_status,
	float* cov3Ds);

/**
 * @brief Wrapper function for gradients computation in a backward pass.
 *
 * @param[in] P                    Number of points to process.
 * @param[in] scales               Array of 3D scales for each point.
 * @param[in] uquats               Array of 3D rotations (unit quaternions) for each point.
 * @param[in] visibility_status    Array indicating the visibility status of each point.
 * @param[in] dL_dcov3Ds           Gradients of the loss with respect to the 3D covariance matrices.
 * @param[out] dL_dscales          Output array for storing the gradients of the loss with respect to scales.
 * @param[out] dL_duquats          Output array for storing the gradients of the loss with respect to rotations.
 */
void computeCov3DBackwardCUDA(
	const int P, 
	const float* scales,
	const float* uquats,
	const bool* visibility_status,
	const float* dL_dcov3Ds,
	float* dL_dscales,
	float* dL_duquats
);

#endif