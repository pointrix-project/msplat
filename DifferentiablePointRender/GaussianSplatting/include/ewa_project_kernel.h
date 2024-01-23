
#ifndef CUDA_EWA_PROJECT_KERNEL_H_INCLUDED
#define CUDA_EWA_PROJECT_KERNEL_H_INCLUDED

void EWAProjectForwardCUDA(
    int P,
    const float* xyz,
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
);

// void EWAProjectBackwardCUDA(
//     int P,
//     const float* xyz,
//     const float* cov3d,
//     const float* viewmat,
//     const float* projmat,
//     const float* camparam,
//     const int W, const int H,
//     const bool* visibility_status,
//     const float* cov2d,
//     float* dL_dxyz,
//     float* dL_dcov3d
// );

#endif

