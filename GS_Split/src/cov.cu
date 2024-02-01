#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ void compute_cov3d_forward(const glm::vec3 scale, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ void compute_cov3_backward(const glm::vec3 scale, const glm::vec4 rotation, const float* dL_dcov3D, glm::vec3& dL_dscale, float4& dL_drotation)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rotation;// / glm::length(rotation);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	dL_dscale.x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale.y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale.z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	dL_drotation = { dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };  //dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

__global__ void computeCov3DForwardCUDAKernel(
    const int P,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const bool* visibility_filter,
    float* cov3Ds)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_filter[idx])
		return;

    compute_cov3d_forward(scales[idx], rotations[idx], cov3Ds + 6 * idx);
}


__global__ void computeCov3DBackwardCUDAKernel(
    const int P,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const bool* visibility_filter,
    const float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    float4* dL_drotations
)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_filter[idx])
		return;

    compute_cov3_backward(scales[idx], rotations[idx], dL_dcov3Ds + 6 * idx, dL_dscales[idx], dL_drotations[idx]);
}

void computeCov3DForwardCUDA(
	const int P, 
	const float* scales,
	const float* rotations,
	const bool* visibility_filter,
	float* cov3Ds)
{
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	computeCov3DForwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
		P,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
        visibility_filter,
		cov3Ds
    );
}

void computeCov3DBackwardCUDA(
	const int P, 
	const float* scales,
	const float* rotations,
	const bool* visibility_filter,
	const float* dL_dcov3Ds,
	float* dL_dscales,
	float* dL_drotations
)
{
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	computeCov3DBackwardCUDAKernel <<<(P + 255) / 256, 256 >>> (
		P,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
        visibility_filter,
		dL_dcov3Ds,
        (glm::vec3*)dL_dscales,
        (float4*)dL_drotations
    );
}