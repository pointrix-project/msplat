
/**
 * @file compute_cov3d.cu
 * @author Jian Gao
 * @brief CUDA kernel for transforming scaling and rotation to a covariance matrix in 3D space.
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <compute_cov3d_kernel.h>

namespace cg = cooperative_groups;

/**
 * @brief Helper function: transform a scaling vector to matrix
 * 
 * @param[in] scale     Scaling vector
 * @return Scaling Matrix 
 */
__device__ inline glm::mat3 scale_vector_to_matrix(const glm::vec3 scale)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	return S;
}

/**
 * @brief Helper function: transform a unit quaternion to rotmatrix
 * 
 * @param[in] uquat    Unit quaternion
 * @return Rotation matrix
 */
__device__ inline glm::mat3 unit_quaternion_to_rotmatrix(const glm::vec4 uquat)
{
    float r = uquat.x;
	float x = uquat.y;
	float y = uquat.z;
	float z = uquat.w;

    return glm::mat3(
		1.f - 2.f * (y * y + z * z), 
        2.f * (x * y - r * z), 
        2.f * (x * z + r * y),
		2.f * (x * y + r * z), 
        1.f - 2.f * (x * x + z * z), 
        2.f * (y * z - r * x),
		2.f * (x * z - r * y), 
        2.f * (y * z + r * x), 
        1.f - 2.f * (x * x + y * y)
	);
}


/**
 * @brief Forward process on the device: converting the scaling and rotation 
 *        of a 3D Gaussian into a covariance matrix.
 * 
 * @param[in] scale    Scaling vector
 * @param[in] quat     Unit quaternion
 * @param[out] cov3D   Covariance vector, the upper right part of the covariance matrix
 */
__device__ void compute_cov3d_forward(
	const glm::vec3 scale, 
	const glm::vec4 quat, 
	float* cov3D)
{
	glm::mat3 S = scale_vector_to_matrix(scale);
	glm::mat3 R = unit_quaternion_to_rotmatrix(quat);
	glm::mat3 M = S * R;

	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0]; cov3D[1] = Sigma[0][1]; 
	cov3D[2] = Sigma[0][2]; cov3D[3] = Sigma[1][1]; 
	cov3D[4] = Sigma[1][2]; cov3D[5] = Sigma[2][2];
}

/**
 * @brief Backward process on the device: converting the scaling and rotation 
 *        of a 3D Gaussian into a covariance matrix.
 * 
 * @param[in] scale       Scaling vector
 * @param[in] quat        Unit quaternion
 * @param[in] dL_dcov3D   loss gradient w.r.t. covariance vector
 * @param[out] dL_dscale  loss gradient w.r.t. scale vector
 * @param[out] dL_dquat   loss gradient w.r.t. unit quaternion
 */
__device__ void compute_cov3_backward(
	const glm::vec3 scale, 
	const glm::vec4 quat, 
	const float* dL_dcov3D, 
	glm::vec3& dL_dscale, 
	glm::vec4& dL_dquat)
{
	glm::mat3 S = scale_vector_to_matrix(scale);
	glm::mat3 R = unit_quaternion_to_rotmatrix(quat);
	glm::mat3 M = S * R;

	// Convert covariance loss gradients from vector to matrix
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	// Loss gradient w.r.t. scale
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	dL_dscale.x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale.y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale.z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale.x;
	dL_dMt[1] *= scale.y;
	dL_dMt[2] *= scale.z;

	// Loss gradients w.r.t. unit quaternion
    float r = quat.x;
	float x = quat.y;
	float y = quat.z;
	float z = quat.w;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) +
			  2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
			  2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) +
			  2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
			  2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) -
			  4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) +
			  2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
			  2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) -
			  4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) +
			  2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
			  2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) -
			  4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
}

/**
 * @brief CUDA kernel for computing 3D covariance matrices in a forward pass. 
 * 
 * @param[in] P                   Number of points to process.
 * @param[in] scales              Array of 3D scales for each point.
 * @param[in] rotations           Array of 3D rotations (quaternions) for each point.
 * @param[in] visibility_status   Array indicating the visibility status of each point.
 * @param[out] cov3Ds             Output array for storing the computed 3D covariance vectors. 
 */
__global__ void computeCov3DForwardCUDAKernel(
    const int P,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const bool* visibility_status,
    float* cov3Ds)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_status[idx])
		return;

    compute_cov3d_forward(
		scales[idx], 
		rotations[idx], 
		cov3Ds + 6 * idx
	);
}

/**
 * @brief CUDA kernel for computing gradients in a backward pass of 3D covariance computation.
 *
 * @param[in] P                    Number of points to process.
 * @param[in] scales               Array of 3D scales for each point.
 * @param[in] rotations            Array of 3D rotations (quaternions) for each point.
 * @param[in] visibility_status    Array indicating the visibility status of each point.
 * @param[in] dL_dcov3Ds           Gradients of the loss with respect to the 3D covariance matrices.
 * @param[out] dL_dscales          Output array for storing the gradients of the loss with respect to scales.
 * @param[out] dL_drotations       Output array for storing the gradients of the loss with respect to rotations.
 */
__global__ void computeCov3DBackwardCUDAKernel(
    const int P,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const bool* visibility_status,
    const float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drotations)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_status[idx])
		return;

    compute_cov3_backward(
		scales[idx], 
		rotations[idx], 
		dL_dcov3Ds + 6 * idx, 
		dL_dscales[idx], 
		dL_drotations[idx]
	);
}

/**
 * @brief Wrapper function for launching the CUDA kernel to compute 3D covariance matrices in a forward pass.
 *
 * @param P                   Number of points to process.
 * @param scales              Array of 3D scales for each point.
 * @param rotations           Array of 3D rotations for each point.
 * @param visibility_status   Array indicating the visibility status of each point.
 * @param cov3Ds              Output array for storing the computed 3D covariance matrices.
 */
void computeCov3DForwardCUDA(
	const int P, 
	const float* scales, 
	const float* rotations, 
	const bool* visibility_status,
	float* cov3Ds)
{
	computeCov3DForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
		P, 
		(glm::vec3*)scales, 
		(glm::vec4*)rotations, 
		visibility_status,
		cov3Ds
	);
}

/**
 * @brief Wrapper function for launching the CUDA kernel to compute gradients in a backward pass
 *        of 3D covariance computation.
 *
 * @param P                   Number of points to process.
 * @param scales              Array of 3D scales for each point.
 * @param rotations           Array of 3D rotations for each point.
 * @param visibility_status   Array indicating the visibility status of each point.
 * @param dL_dcov3Ds          Gradients of the loss with respect to the 3D covariance matrices.
 * @param dL_dscales          Output array for storing the gradients of the loss with respect to scales.
 * @param dL_drotations       Output array for storing the gradients of the loss with respect to rotations.
 */
void computeCov3DBackwardCUDA(
	const int P, 
	const float* scales,
	const float* rotations,
	const bool* visibility_status,
	const float* dL_dcov3Ds,
	float* dL_dscales,
	float* dL_drotations)
{
	computeCov3DBackwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
		P, 
		(glm::vec3*)scales, 
		(glm::vec4*)rotations, 
		visibility_status,
		dL_dcov3Ds, 
		(glm::vec3*)dL_dscales, 
		(glm::vec4*)dL_drotations
	);
}