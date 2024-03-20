/**
 * @file compute_sh.cu
 * @brief
 */

// https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#cite_note-Chisholm1976-4

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/torch.h>
#include <utils.h>

namespace cg = cooperative_groups;

__device__ const float SH_C0 = 0.28209479177387814f;

__device__ const float SH_C1[] = {
    -0.4886025119029199f, 0.4886025119029199f, -0.4886025119029199f};

__device__ const float SH_C2[] = {1.0925484305920792f,
                                  -1.0925484305920792f,
                                  0.31539156525252005f,
                                  -1.0925484305920792f,
                                  0.5462742152960396f};

__device__ const float SH_C3[] = {-0.5900435899266435f,
                                  2.890611442640554f,
                                  -0.4570457994644658f,
                                  0.3731763325901154f,
                                  -0.4570457994644658f,
                                  1.445305721320277f,
                                  -0.5900435899266435f};

__device__ const float SH_C4[] = {2.5033429417967046f,
                                  -1.7701307697799304f,
                                  0.9461746957575601f,
                                  -0.6690465435572892f,
                                  0.10578554691520431f,
                                  -0.6690465435572892f,
                                  0.47308734787878004f,
                                  -1.7701307697799304f,
                                  0.6258357354491761f};

__device__ const float SH_C5[] = {-0.3281910284200851f,
                                  1.0378311574405208f,
                                  -0.2446191497176252f,
                                  1.198384196243331f,
                                  -0.22647332559784847f,
                                  0.1169503224534236f,
                                  -0.22647332559784847f,
                                  1.198384196243331f,
                                  -0.2446191497176252f,
                                  1.0378311574405208f,
                                  -0.3281910284200851f};

__device__ const float SH_C6[] = {0.34159205259595715f,
                                  -1.183309581115876f,
                                  0.2522824503643621f,
                                  -0.46060262975746175f,
                                  0.23030131487873087f,
                                  -0.2913106812593657f,
                                  0.06356920226762842f,
                                  -0.2913106812593657f,
                                  0.23030131487873087f,
                                  -0.46060262975746175f,
                                  0.2522824503643621f,
                                  -1.183309581115876f,
                                  0.34159205259595715f};

__device__ const float SH_C7[] = {-0.3535813662622981f,
                                  1.32298033090095f,
                                  -0.2594577893601302f,
                                  0.5189155787202604f,
                                  -0.07822946693114702f,
                                  0.11063317311124565f,
                                  -0.04516580379125865f,
                                  0.06828427691200495f,
                                  -0.04516580379125865f,
                                  0.11063317311124565f,
                                  -0.07822946693114702f,
                                  0.5189155787202604f,
                                  -0.2594577893601302f,
                                  1.32298033090095f,
                                  -0.3535813662622981f};

__device__ const float SH_C8[] = {
    0.36446333008741494f,
    -1.4578533203496598f, 
    0.2661663830297713f, 
    -1.724955311049054f,
    0.23920826237966533f,
    -0.6176330776477721f,
    0.22807612921745474f,
    -0.054520622949389974f,
    0.009086770491564996f,
    -0.054520622949389974f,
    0.22807612921745474f,
    -0.6176330776477721f, 
    0.23920826237966533f,
    -1.724955311049054f,
    0.2661663830297713f,
    -1.4578533203496598f,
    0.36446333008741494f
};

__device__ const float SH_C9[] = [
    -0.37445047592659414,
    1.5886588244773487,
    -0.2724527406720266,
    0.6292026526737834,
    -0.2436891395195093,
    2.0388496193645866,
    -0.23085426000809722,
    0.30225917742574926,
    -0.03222093657611365,
    0.009606427264386591,
    -0.03222093657611365,
    0.30225917742574926,
    -0.23085426000809722,
    2.0388496193645866,
    -0.2436891395195093,
    0.6292026526737834, 
    -0.2724527406720266, 
    1.5886588244773487,
    -0.3744504759265941,
]

__device__ const float SH_C10[] = [
    0.383697559110995,
    -1.7159476499458572,
    0.2783634663602092,
    -0.6818484556149027,
    0.08268627068130938,
    -0.1479136976394845,
    0.23387209083912108,
    -0.3307450827252375,
    0.032432223670016275,
    -0.037449506132604095,
    0.005049690376783604,
    -0.037449506132604095,
    0.032432223670016275,
    -0.3307450827252375,
    0.23387209083912108,
    -0.1479136976394845,
    0.08268627068130938,
    -0.6818484556149027,
    0.2783634663602092,
    -1.7159476499458572,
    0.383697559110995
]

__forceinline__ __device__ float evaluateSH0Forward(const float *sh) {
    return SH_C0 * sh[0];
}

__forceinline__ __device__ float evaluateSH1Forward(const float *sh,
                                                    const float3 dir) {
    return SH_C1[0] * dir.y * sh[1] + SH_C1[1] * dir.z * sh[2] +
           SH_C1[2] * dir.x * sh[3];
}

__forceinline__ __device__ float evaluateSH2Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = SH_C2[0] * xy * sh[4];
    result += SH_C2[1] * yz * sh[5];
    result += SH_C2[2] * (2.0f * z2 - x2 - y2) * sh[6];
    result += SH_C2[3] * xz * sh[7];
    result += SH_C2[4] * (x2 - y2) * sh[8];

    return result;
}

__forceinline__ __device__ float evaluateSH3Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = SH_C3[0] * y * (3.0f * x2 - y2) * sh[9];
    result += SH_C3[1] * xy * z * sh[10];
    result += SH_C3[2] * y * (4.0f * z2 - x2 - y2) * sh[11];
    result += SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2) * sh[12];
    result += SH_C3[4] * x * (4.0f * z2 - x2 - y2) * sh[13];
    result += SH_C3[5] * z * (x2 - y2) * sh[14];
    result += SH_C3[6] * x * (x2 - 3.0f * y2) * sh[15];

    return result;
}

__forceinline__ __device__ float evaluateSH4Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = SH_C4[0] * xy * (x2 - y2) * sh[16];
    result += SH_C4[1] * yz * (3 * x2 - y2) * sh[17];
    result += SH_C4[2] * xy * (7 * z2 - 1) * sh[18];
    result += SH_C4[3] * yz * (7 * z2 - 3) * sh[19];
    result += SH_C4[4] * (z2 * (35 * z2 - 30) + 3) * sh[20];
    result += SH_C4[5] * xz * (7 * z2 - 3) * sh[21];
    result += SH_C4[6] * (x2 - y2) * (7 * z2 - 1) * sh[22];
    result += SH_C4[7] * xz * (x2 - 3 * y2) * sh[23];
    result += SH_C4[8] * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) * sh[24];

    return result;
}

__forceinline__ __device__ float evaluateSH5Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    float result =
        SH_C5[0] * (10 * x3 * xy - 20 * x2 * y3 + 2 * y3 * y2) * sh[25];
    result += SH_C5[1] * (8 * x3 * yz - 8 * xy * y2 * z) * sh[26];
    result += SH_C5[2] *
              (54 * x2 * y * z2 - 6 * x2 * y - 18 * y3 * z2 + 2 * y3) * sh[27];
    result += SH_C5[3] * (12 * xy * z3 - 4 * x * yz) * sh[28];
    result += SH_C5[4] * (42 * yz * z3 - 28 * y * z2 + 2 * y) * sh[29];
    result += SH_C5[5] * (63 * z2 * z3 - 70 * z3 + 15 * z) * sh[30];
    result += SH_C5[6] * (42 * x * z2 * z2 - 28 * x * z2 + 2 * x) * sh[31];
    result += SH_C5[7] * (6 * x2 * z3 - 2 * x2 * z - 6 * y2 * z3 + 2 * y2 * z) *
              sh[32];
    result += SH_C5[8] *
              (18 * x3 * z2 - 2 * x3 - 54 * x * y2 * z2 + 6 * x * y2) * sh[33];
    result += SH_C5[9] *
              (2 * x2 * x2 * z - 12 * x2 * y2 * z + 2 * y2 * y2 * z) * sh[34];
    result += SH_C5[10] * (2 * x3 * x2 - 20 * x3 * y2 + 10 * xy * y3) * sh[35];

    return result;
}

__forceinline__ __device__ float evaluateSH6Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    float result =
        SH_C6[0] * (12 * x3 * x2 * y - 40 * x3 * y3 + 12 * xy * y4) * sh[36];
    result +=
        SH_C6[1] * (10 * x4 * yz - 20 * x2 * y2 * yz + 2 * y4 * yz) * sh[37];
    result +=
        SH_C6[2] *
        (88 * x2 * xy * z2 - 8 * x2 * xy - 88 * xy * y2 * z2 + 8 * xy * y2) *
        sh[38];
    result += SH_C6[3] * 2 * yz *
              (33 * x2 * z2 - 9 * x2 - 11 * y2 * z2 + 3 * y2) * sh[39];
    result += SH_C6[4] * 4 * xy * (33 * z4 - 18 * z2 + 1) * sh[40];
    result += SH_C6[5] * 2 * yz * (33 * z4 - 30 * z2 + 5) * sh[41];
    result += SH_C6[6] * (231 * z3 * z3 - 315 * z4 + 105 * z2 - 5) * sh[42];
    result += SH_C6[7] * (66 * xz * z4 - 60 * xz * z2 + 10 * xz) * sh[43];
    result += SH_C6[8] *
              (66 * x2 * z4 - 36 * x2 * z2 + 2 * x2 - 66 * y2 * z4 +
               36 * y2 * z2 - 2 * y2) *
              sh[44];
    result += SH_C6[9] *
              (22 * x3 * z3 - 6 * x2 * xz - 66 * xy * yz * z2 + 18 * xy * yz) *
              sh[45];
    result += SH_C6[10] *
              (22 * x4 * z2 - 2 * x4 - 132 * x2 * y2 * z2 + 12 * x2 * y2 +
               22 * y4 * z2 - 2 * y4) *
              sh[46];
    result += SH_C6[11] *
              (2 * x4 * xz - 20 * x2 * xy * yz + 10 * xy * y2 * yz) * sh[47];
    result += SH_C6[12] *
              (2 * x4 * x2 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y3 * y3) *
              sh[48];

    return result;
}

__forceinline__ __device__ float evaluateSH7Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    float result = SH_C7[0] * 2 * y *
                   (7 * x3 * x3 - 35 * x4 * y2 + 21 * x2 * y4 - y3 * y3) *
                   sh[49];
    result += SH_C7[1] * 4 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4) * sh[50];
    result += SH_C7[2] * 2 * y *
              (65 * x4 * z2 - 5 * x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
               13 * y4 * z2 - y4) *
              sh[51];
    result += SH_C7[3] * 8 * xyz *
              (13 * x2 * z2 - 3 * x2 - 13 * y2 * z2 + 3 * y2) * sh[52];
    result += SH_C7[4] * 2 * y *
              (429 * x2 * z4 - 198 * x2 * z2 + 9 * x2 - 143 * y2 * z4 +
               66 * y2 * z2 - 3 * y2) *
              sh[53];
    result += SH_C7[5] * 4 * xyz * (143 * z4 - 110 * z2 + 15) * sh[54];
    result +=
        SH_C7[6] * 2 * y * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5) * sh[55];
    result +=
        SH_C7[7] * z * (429 * z3 * z3 - 693 * z4 + 315 * z2 - 35) * sh[56];
    result +=
        SH_C7[8] * 2 * x * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5) * sh[57];
    result += SH_C7[9] * 2 * z *
              (143 * x2 * z4 - 110 * x2 * z2 + 15 * x2 - 143 * y2 * z4 +
               110 * y2 * z2 - 15 * y2) *
              sh[58];
    result += SH_C7[10] * 2 * x *
              (143 * x2 * z4 - 66 * x2 * z2 + 3 * x2 - 429 * y2 * z4 +
               198 * y2 * z2 - 9 * y2) *
              sh[59];
    result += SH_C7[11] * 2 * z *
              (13 * x4 * z2 - 3 * x4 - 78 * x2 * y2 * z2 + 18 * x2 * y2 +
               13 * y4 * z2 - 3 * y4) *
              sh[60];
    result += SH_C7[12] * 2 * x *
              (13 * x4 * z2 - x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
               65 * y4 * z2 - 5 * y4) *
              sh[61];
    result += SH_C7[13] * 2 * z *
              (x3 * x3 - 15 * x4 * y2 + 15 * x2 * y4 - y3 * y3) * sh[62];
    result += SH_C7[14] * 2 * x *
              (x3 * x3 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y3 * y3) * sh[63];

    return result;
}

__forceinline__ __device__ float evaluateSH8Forward(const float *sh,
                                                    const float3 dir) {
    
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;

    float result = SH_C8[0] * 16 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6) * sh[64];
    result += SH_C8[1] * 2 * yz * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6)* sh[65];
    result += SH_C8[2] * 4 * xy*(45 * x4 * z2 - 3 * x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 + 45 * y4 * z2 - 3 * y4) * sh[66];
    result += SH_C8[3] * 2 * yz * (25 * x4 * z2 - 5 * x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 + 5 * y4 * z2 - y4) * sh[67];
    result += SH_C8[4] *8 * xy * (65 * x2 * z4 - 26 * x2 * z2 + x2 - 65 * y2 * z4 + 26 * y2 * z2 - y2) * sh[68];
    result += SH_C8[5] *2 * yz * (117 * x2 * z4 - 78 * x2 * z2 + 9 * x2 - 39 * y2 * z4 + 26 * y2 * z2 - 3 * y2) * sh[69];
    result += SH_C8[6] *4 * xy * (143 * z6 - 143 * z4 + 33 * z2 - 1) * sh[70];  
    result += SH_C8[7] *2 * yz * (715 * z6 - 1001 * z4 + 385 * z2 - 35) * sh[71];
    result += SH_C8[8] *(6435 * z4 * z4 - 12012 * z6 + 6930 * z4 - 1260 * z2 + 35) * sh[72];
    result += SH_C8[9] * 2 * xz * (715 * z6 - 1001 * z4 + 385 * z2 - 35) * sh[73];
    result += SH_C8[10] *(286 * x2 * z6 - 286 * x2 * z4 + 66 * x2 * z2 - 2 * x2 - 286 * y2 * z6 + 286 * y2 * z4 - 66 * y2*z2 + 2*y2) * sh[74];
    result += SH_C8[11] * 2* xz*(39*x2*z4 - 26*x2*z2 + 3*x2 - 117*y2*z4 + 78*y2*z2 - 9*y2) * sh[75];
    result += SH_C8[12] * (130*x4*z4 - 52*x4*z2 + 2*x4 - 780*x2*y2*z4 + 312*x2*y2*z2 - 12*x2*y2 + 130*y4*z4 - 52*y4*z2 + 2*y4) * sh[76];
    result += SH_C8[13] * 2 * xz * (5*x4*z2 - x4 - 50*x2*y2*z2 + 10*x2*y2 + 25*y4*z2 - 5*y4) * sh[77];
    result += SH_C8[14] * (30*x6*z2 - 2*x6 - 450*x4*y2*z2 + 30*x4*y2 + 450*x2*y4*z2 - 30*x2*y4 - 30*y6*z2 + 2*y6) * sh[78];
    result += SH_C8[15] * 2*xz*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6) * sh[79];
    result += SH_C8[16] * (2* x4 * x4 - 56*x6*y2 + 140*x4*y4 - 56*x2*y6 + 2*y4 * y4) * sh[80];

    return result;
}

__forceinline__ __device__ float evaluateSH9Forward(const float *sh,
                                                    const float3 dir) {
    
    return 0;
}


__forceinline__ __device__ float evaluateSH8Forward(const float *sh,
                                                    const float3 dir) {

    return 0;
}


__forceinline__ __device__ void evaluateSH0Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    dL_dsh[0] = SH_C0 * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH1Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    dL_dsh[1] = SH_C1[0] * dir.y * dL_dvalue;
    dL_dsh[2] = SH_C1[1] * dir.z * dL_dvalue;
    dL_dsh[3] = SH_C1[2] * dir.x * dL_dvalue;

    float3 dvalue_ddir = {SH_C1[2] * sh[3], SH_C1[0] * sh[1], SH_C1[1] * sh[2]};

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH2Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    dL_dsh[4] = SH_C2[0] * xy;
    dL_dsh[5] = SH_C2[1] * yz;
    dL_dsh[6] = SH_C2[2] * (2.0f * z2 - x2 - y2);
    dL_dsh[7] = SH_C2[3] * xz;
    dL_dsh[8] = SH_C2[4] * (x2 - y2);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C2[0] * sh[4] * y;
    dvalue_ddir.x += SH_C2[2] * sh[6] * (-2 * x);
    dvalue_ddir.x += SH_C2[3] * sh[7] * z;
    dvalue_ddir.x += SH_C2[4] * sh[8] * 2 * x;

    dvalue_ddir.y += SH_C2[0] * sh[4] * x;
    dvalue_ddir.y += SH_C2[1] * sh[5] * z;
    dvalue_ddir.y += SH_C2[2] * sh[6] * (-2 * y);
    dvalue_ddir.y += SH_C2[4] * sh[8] * (-2 * y);

    dvalue_ddir.z += SH_C2[1] * sh[5] * y;
    dvalue_ddir.z += SH_C2[2] * sh[6] * 4 * z;
    dvalue_ddir.z += SH_C2[3] * sh[7] * x;

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH3Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    dL_dsh[9] = SH_C3[0] * y * (3.0f * x2 - y2);
    dL_dsh[10] = SH_C3[1] * xy * z;
    dL_dsh[11] = SH_C3[2] * y * (4.0f * z2 - x2 - y2);
    dL_dsh[12] = SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2);
    dL_dsh[13] = SH_C3[4] * x * (4.0f * z2 - x2 - y2);
    dL_dsh[14] = SH_C3[5] * z * (x2 - y2);
    dL_dsh[15] = SH_C3[6] * x * (x2 - 3.0f * y2);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C3[0] * sh[9] * 6 * xy;
    dvalue_ddir.x += SH_C3[1] * sh[10] * yz;
    dvalue_ddir.x += SH_C3[2] * sh[11] * (-2 * xy);
    dvalue_ddir.x += SH_C3[3] * sh[12] * (-6 * xz);
    dvalue_ddir.x += SH_C3[4] * sh[13] * (4 * z2 - 3 * x2 - y2);
    dvalue_ddir.x += SH_C3[5] * sh[14] * 2 * xz;
    dvalue_ddir.x += SH_C3[6] * sh[15] * (3 * x2 - 3 * y2);

    dvalue_ddir.y += SH_C3[0] * sh[9] * (3 * x2 - 3 * y2);
    dvalue_ddir.y += SH_C3[1] * sh[10] * xz;
    dvalue_ddir.y += SH_C3[2] * sh[11] * (4 * z2 - x2 - 3 * y2);
    dvalue_ddir.y += SH_C3[3] * sh[12] * (-6.0 * yz);
    dvalue_ddir.y += SH_C3[4] * sh[13] * (-2.0 * xy);
    dvalue_ddir.y += SH_C3[5] * sh[14] * (-2.0 * yz);
    dvalue_ddir.y += SH_C3[6] * sh[15] * (-6.0 * xy);

    dvalue_ddir.z += SH_C3[1] * sh[10] * xy;
    dvalue_ddir.z += SH_C3[2] * sh[11] * 8 * yz;
    dvalue_ddir.z += SH_C3[3] * sh[12] * (6 * z2 - 3 * x2 - 3 * y2);
    dvalue_ddir.z += SH_C3[4] * sh[13] * 8 * xz;
    dvalue_ddir.z += SH_C3[5] * sh[14] * (x2 - y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH4Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    dL_dsh[16] = SH_C4[0] * xy * (x2 - y2);
    dL_dsh[17] = SH_C4[1] * yz * (3 * x2 - y2);
    dL_dsh[18] = SH_C4[2] * xy * (7 * z2 - 1);
    dL_dsh[19] = SH_C4[3] * yz * (7 * z2 - 3);
    dL_dsh[20] = SH_C4[4] * (z2 * (35 * z2 - 30) + 3);
    dL_dsh[21] = SH_C4[5] * xz * (7 * z2 - 3);
    dL_dsh[22] = SH_C4[6] * (x2 - y2) * (7 * z2 - 1);
    dL_dsh[23] = SH_C4[7] * xz * (x2 - 3 * y2);
    dL_dsh[24] = SH_C4[8] * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2));

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C4[0] * sh[16] * (3 * x2 * y - y3);
    dvalue_ddir.x += SH_C4[1] * sh[17] * 6 * xyz;
    dvalue_ddir.x += SH_C4[2] * sh[18] * (7 * y * z2 - y);
    dvalue_ddir.x += SH_C4[5] * sh[21] * (7 * z3 - 3 * z);
    dvalue_ddir.x += SH_C4[6] * sh[22] * (14 * x * z2 - 2 * x);
    dvalue_ddir.x += SH_C4[7] * sh[23] * (3 * x2 * z - 3 * y2 * z);
    dvalue_ddir.x += SH_C4[8] * sh[24] * (4 * x3 - 6 * x * y2 - 6 * x * y2);

    dvalue_ddir.y += SH_C4[0] * sh[16] * (x3 - 3 * x * y2);
    dvalue_ddir.y += SH_C4[1] * sh[17] * (3 * x2 * z - 3 * y2 * z);
    dvalue_ddir.y += SH_C4[2] * sh[18] * (7 * x * z2 - x);
    dvalue_ddir.y += SH_C4[3] * sh[19] * (7 * z3 - 3 * z);
    dvalue_ddir.y += SH_C4[6] * sh[22] * (-14 * y * z2 + 2 * y);
    dvalue_ddir.y += SH_C4[7] * sh[23] * (-6 * xyz);
    dvalue_ddir.y += SH_C4[8] * sh[24] * (-12 * x2 * y + 4 * y3);

    dvalue_ddir.z += SH_C4[1] * sh[17] * (3 * x2 * y - y3);
    dvalue_ddir.z += SH_C4[2] * sh[18] * 14 * xyz;
    dvalue_ddir.z += SH_C4[3] * sh[19] * (21 * y * z2 - 3 * y);
    dvalue_ddir.z += SH_C4[4] * sh[20] * (140 * z3 - 60 * z);
    dvalue_ddir.z += SH_C4[5] * sh[21] * (21 * x * z2 - 3 * x);
    dvalue_ddir.z += SH_C4[6] * sh[22] * (14 * x2 * z - 14 * y2 * z);
    dvalue_ddir.z += SH_C4[7] * sh[23] * (x3 - 3 * x * y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH5Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    dL_dsh[25] = SH_C5[0] * (10 * x3 * xy - 20 * x2 * y3 + 2 * y3 * y2);
    dL_dsh[26] = SH_C5[1] * (8 * x3 * yz - 8 * xy * y2 * z);
    dL_dsh[27] =
        SH_C5[2] * (54 * x2 * y * z2 - 6 * x2 * y - 18 * y3 * z2 + 2 * y3);
    dL_dsh[28] = SH_C5[3] * (12 * xy * z3 - 4 * x * yz);
    dL_dsh[29] = SH_C5[4] * (42 * yz * z3 - 28 * y * z2 + 2 * y);
    dL_dsh[30] = SH_C5[5] * (63 * z2 * z3 - 70 * z3 + 15 * z);
    dL_dsh[31] = SH_C5[6] * (42 * x * z2 * z2 - 28 * x * z2 + 2 * x);
    dL_dsh[32] =
        SH_C5[7] * (6 * x2 * z3 - 2 * x2 * z - 6 * y2 * z3 + 2 * y2 * z);
    dL_dsh[33] =
        SH_C5[8] * (18 * x3 * z2 - 2 * x3 - 54 * x * y2 * z2 + 6 * x * y2);
    dL_dsh[34] =
        SH_C5[9] * (2 * x2 * x2 * z - 12 * x2 * y2 * z + 2 * y2 * y2 * z);
    dL_dsh[35] = SH_C5[10] * (2 * x3 * x2 - 20 * x3 * y2 + 10 * xy * y3);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C5[0] * sh[25] * 40 * xy * (x2 - y2);
    dvalue_ddir.x += SH_C5[1] * sh[26] * 8 * yz * (3 * x2 - y2);
    dvalue_ddir.x += SH_C5[2] * sh[27] * 12 * xy * (9 * z2 - 1);
    dvalue_ddir.x += SH_C5[3] * sh[28] * 4 * yz * (3 * z2 - 1);
    dvalue_ddir.x += SH_C5[6] * sh[31] * (42 * z2 * z2 - 28 * z2 + 2);
    dvalue_ddir.x += SH_C5[7] * sh[32] * 4 * xz * (3 * z2 - 1);
    dvalue_ddir.x +=
        SH_C5[8] * sh[33] * (54 * x2 * z2 - 6 * x2 - 54 * y2 * z2 + 6 * y2);
    dvalue_ddir.x += SH_C5[9] * sh[34] * 8 * xz * (x2 - 3 * y2);
    dvalue_ddir.x +=
        SH_C5[10] * sh[35] * (10 * x2 * x2 - 60 * x2 * y2 + 10 * y2 * y2);

    dvalue_ddir.y +=
        SH_C5[0] * sh[25] * (10 * x2 * x2 - 60 * x2 * y2 + 10 * y2 * y2);
    dvalue_ddir.y += SH_C5[1] * sh[26] * 8 * xz * (x2 - 3 * y2);
    dvalue_ddir.y +=
        SH_C5[2] * sh[27] * (54 * x2 * z2 - 6 * x2 - 54 * y2 * z2 + 6 * y2);
    dvalue_ddir.y += SH_C5[3] * sh[28] * 4 * xz * (3 * z2 - 1);
    dvalue_ddir.y += SH_C5[4] * sh[29] * (42 * z2 * z2 - 28 * z2 + 2);
    dvalue_ddir.y += SH_C5[7] * sh[32] * 4 * yz * (1 - 3 * z2);
    dvalue_ddir.y += SH_C5[8] * sh[33] * 12 * xy * (1 - 9 * z2);
    dvalue_ddir.y += SH_C5[9] * sh[34] * 8 * yz * (-3 * x2 + y2);
    dvalue_ddir.y += SH_C5[10] * sh[35] * 40 * xy * (-x2 + y2);

    dvalue_ddir.z += SH_C5[1] * sh[26] * 8 * xy * (x2 - y2);
    dvalue_ddir.z += SH_C5[2] * sh[27] * 36 * yz * (3 * x2 - y2);
    dvalue_ddir.z += SH_C5[3] * sh[28] * 4 * xy * (9 * z2 - 1);
    dvalue_ddir.z += SH_C5[4] * sh[29] * 56 * yz * (3 * z2 - 1);
    dvalue_ddir.z += SH_C5[5] * sh[30] * (315 * z2 * z2 - 210 * z2 + 15);
    dvalue_ddir.z += SH_C5[6] * sh[31] * (56 * xz * (3 * z2 - 1));
    dvalue_ddir.z +=
        SH_C5[7] * sh[32] * (18 * x2 * z2 - 2 * x2 - 18 * y2 * z2 + 2 * y2);
    dvalue_ddir.z += SH_C5[8] * sh[33] * 36 * xz * (x2 - 3 * y2);
    dvalue_ddir.z +=
        SH_C5[9] * sh[34] * (2 * x2 * x2 - 12 * x2 * y2 + 2 * y2 * y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH6Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    dL_dsh[36] = SH_C6[0] * (12 * x3 * x2 * y - 40 * x3 * y3 + 12 * xy * y4);
    dL_dsh[37] = SH_C6[1] * (10 * x4 * yz - 20 * x2 * y2 * yz + 2 * y4 * yz);
    dL_dsh[38] = SH_C6[2] * (88 * x2 * xy * z2 - 8 * x2 * xy -
                             88 * xy * y2 * z2 + 8 * xy * y2);
    dL_dsh[39] =
        SH_C6[3] * 2 * yz * (33 * x2 * z2 - 9 * x2 - 11 * y2 * z2 + 3 * y2);
    dL_dsh[40] = SH_C6[4] * 4 * xy * (33 * z4 - 18 * z2 + 1);
    dL_dsh[41] = SH_C6[5] * 2 * yz * (33 * z4 - 30 * z2 + 5);
    dL_dsh[42] = SH_C6[6] * (231 * z3 * z3 - 315 * z4 + 105 * z2 - 5);
    dL_dsh[43] = SH_C6[7] * (66 * xz * z4 - 60 * xz * z2 + 10 * xz);
    dL_dsh[44] = SH_C6[8] * (66 * x2 * z4 - 36 * x2 * z2 + 2 * x2 -
                             66 * y2 * z4 + 36 * y2 * z2 - 2 * y2);
    dL_dsh[45] = SH_C6[9] * (22 * x3 * z3 - 6 * x2 * xz - 66 * xy * yz * z2 +
                             18 * xy * yz);
    dL_dsh[46] = SH_C6[10] * (22 * x4 * z2 - 2 * x4 - 132 * x2 * y2 * z2 +
                              12 * x2 * y2 + 22 * y4 * z2 - 2 * y4);
    dL_dsh[47] =
        SH_C6[11] * (2 * x4 * xz - 20 * x2 * xy * yz + 10 * xy * y2 * yz);
    dL_dsh[48] =
        SH_C6[12] * (2 * x4 * x2 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y3 * y3);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C6[0] * sh[36] * 12 * y * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.x += SH_C6[1] * sh[37] * 40 * xyz * (x2 - y2);
    dvalue_ddir.x +=
        SH_C6[2] * sh[38] * 8 * y * (33 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + y2);
    dvalue_ddir.x += SH_C6[3] * sh[39] * 12 * xyz * (11 * z2 - 3);
    dvalue_ddir.x += SH_C6[4] * sh[40] * 4 * y * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.x += SH_C6[7] * sh[43] * (66 * z3 * z2 - 60 * z3 + 10 * z);
    dvalue_ddir.x += SH_C6[8] * sh[44] * 4 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.x += SH_C6[9] * sh[45] * 6 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.x += SH_C6[10] * sh[46] * 8 * x *
                     (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.x += SH_C6[11] * sh[47] * 10 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.x += SH_C6[12] * sh[48] * 12 * x * (x4 - 10 * x2 * y2 + 5 * y4);

    dvalue_ddir.y += SH_C6[0] * sh[36] * 12 * x * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.y += SH_C6[1] * sh[37] * 10 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.y +=
        SH_C6[2] * sh[38] * 8 * x * (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.y += SH_C6[3] * sh[39] * 6 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.y += SH_C6[4] * sh[40] * 4 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.y += SH_C6[5] * sh[41] * (66 * z3 * z2 - 60 * z3 + 10 * z);
    dvalue_ddir.y += SH_C6[8] * sh[44] * 4 * y * (-33 * z4 + 18 * z2 - 1);
    dvalue_ddir.y += SH_C6[9] * sh[45] * 12 * xyz * (3 - 11 * z2);
    dvalue_ddir.y += SH_C6[10] * sh[46] * 8 * y *
                     (-33 * x2 * z2 + 3 * x2 + 11 * y2 * z2 - y2);
    dvalue_ddir.y += SH_C6[11] * sh[47] * 40 * xyz * (-x2 + y2);
    dvalue_ddir.y +=
        SH_C6[12] * sh[48] * 12 * y * (-5 * x4 + 10 * x2 * y2 - y4);

    dvalue_ddir.z += SH_C6[1] * sh[37] * 2 * y * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.z += SH_C6[2] * sh[38] * 176 * xyz * (x2 - y2);
    dvalue_ddir.z +=
        SH_C6[3] * sh[39] * 6 * y * (33 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + y2);
    dvalue_ddir.z += SH_C6[4] * sh[40] * 48 * xyz * (11 * z2 - 3);
    dvalue_ddir.z += SH_C6[5] * sh[41] * 10 * y * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.z += SH_C6[6] * sh[42] * (1386 * z3 * z2 - 1260 * z3 + 210 * z);
    dvalue_ddir.z += SH_C6[7] * sh[43] * 10 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.z += SH_C6[8] * sh[44] * 24 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.z +=
        SH_C6[9] * sh[45] * 6 * x * (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.z += SH_C6[10] * sh[46] * 44 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.z += SH_C6[11] * sh[47] * 2 * x * (x4 - 10 * x2 * y2 + 5 * y4);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH7Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x5 = x3 * x2, y5 = y3 * y2, z5 = z3 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;

    dL_dsh[49] = SH_C7[0] * 2 * y *
                 (7 * x3 * x3 - 35 * x4 * y2 + 21 * x2 * y4 - y3 * y3);
    dL_dsh[50] = SH_C7[1] * 4 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4);
    dL_dsh[51] = SH_C7[2] * 2 * y *
                 (65 * x4 * z2 - 5 * x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
                  13 * y4 * z2 - y4);
    dL_dsh[52] =
        SH_C7[3] * 8 * xyz * (13 * x2 * z2 - 3 * x2 - 13 * y2 * z2 + 3 * y2);
    dL_dsh[53] = SH_C7[4] * 2 * y *
                 (429 * x2 * z4 - 198 * x2 * z2 + 9 * x2 - 143 * y2 * z4 +
                  66 * y2 * z2 - 3 * y2);
    dL_dsh[54] = SH_C7[5] * 4 * xyz * (143 * z4 - 110 * z2 + 15);
    dL_dsh[55] = SH_C7[6] * 2 * y * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    dL_dsh[56] = SH_C7[7] * z * (429 * z3 * z3 - 693 * z4 + 315 * z2 - 35);
    dL_dsh[57] = SH_C7[8] * 2 * x * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    dL_dsh[58] = SH_C7[9] * 2 * z *
                 (143 * x2 * z4 - 110 * x2 * z2 + 15 * x2 - 143 * y2 * z4 +
                  110 * y2 * z2 - 15 * y2);
    dL_dsh[59] = SH_C7[10] * 2 * x *
                 (143 * x2 * z4 - 66 * x2 * z2 + 3 * x2 - 429 * y2 * z4 +
                  198 * y2 * z2 - 9 * y2);
    dL_dsh[60] = SH_C7[11] * 2 * z *
                 (13 * x4 * z2 - 3 * x4 - 78 * x2 * y2 * z2 + 18 * x2 * y2 +
                  13 * y4 * z2 - 3 * y4);
    dL_dsh[61] = SH_C7[12] * 2 * x *
                 (13 * x4 * z2 - x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
                  65 * y4 * z2 - 5 * y4);
    dL_dsh[62] =
        SH_C7[13] * 2 * z * (x3 * x3 - 15 * x4 * y2 + 15 * x2 * y4 - y3 * y3);
    dL_dsh[63] = SH_C7[14] * 2 * x *
                 (x3 * x3 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y3 * y3);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x +=
        SH_C7[0] * sh[49] * (84 * x5 * y - 280 * x3 * y3 + 84 * x * y5);
    dvalue_ddir.x += SH_C7[1] * sh[50] * 12 * yz * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.x +=
        SH_C7[2] * sh[51] * 40 * xy * (13 * x2 * z2 - x2 - 13 * y2 * z2 + y2);
    dvalue_ddir.x += SH_C7[3] * sh[52] * 8 * yz *
                     (39 * x2 * z2 - 9 * x2 - 13 * y2 * z2 + 3 * y2);
    dvalue_ddir.x += SH_C7[4] * sh[53] * 12 * xy * (143 * z4 - 66 * z2 + 3);
    dvalue_ddir.x += SH_C7[5] * sh[54] * 4 * yz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.x += SH_C7[8] * sh[57] * (858 * z6 - 990 * z4 + 270 * z2 - 10);
    dvalue_ddir.x += SH_C7[9] * sh[58] * 4 * xz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.x += SH_C7[10] * sh[59] *
                     (858 * x2 * z4 - 396 * x2 * z2 + 18 * x2 - 858 * y2 * z4 +
                      396 * y2 * z2 - 18 * y2);
    dvalue_ddir.x += SH_C7[11] * sh[60] * 8 * xz *
                     (13 * x2 * z2 - 3 * x2 - 39 * y2 * z2 + 9 * y2);
    dvalue_ddir.x += SH_C7[12] * sh[61] *
                     (130 * x4 * z2 - 10 * x4 - 780 * x2 * y2 * z2 +
                      60 * x2 * y2 + 130 * y4 * z2 - 10 * y4);
    dvalue_ddir.x +=
        SH_C7[13] * sh[62] * 12 * xz * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.x += SH_C7[14] * sh[63] *
                     (14 * x6 - 210 * x4 * y2 + 210 * x2 * y4 - 14 * y6);

    dvalue_ddir.y +=
        SH_C7[0] * sh[49] * (14 * x6 - 210 * x4 * y2 + 210 * x2 * y4 - 14 * y6);
    dvalue_ddir.y += SH_C7[1] * sh[50] * 12 * xz * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.y += SH_C7[2] * sh[51] *
                     (130 * x4 * z2 - 10 * x4 - 780 * x2 * y2 * z2 +
                      60 * x2 * y2 + 130 * y4 * z2 - 10 * y4);
    dvalue_ddir.y += SH_C7[3] * sh[52] * 8 * xz *
                     (13 * x2 * z2 - 3 * x2 - 39 * y2 * z2 + 9 * y2);
    dvalue_ddir.y += SH_C7[4] * sh[53] *
                     (858 * x2 * z4 - 396 * x2 * z2 + 18 * x2 - 858 * y2 * z4 +
                      396 * y2 * z2 - 18 * y2);
    dvalue_ddir.y += SH_C7[5] * sh[54] * 4 * xz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.y += SH_C7[6] * sh[55] * (858 * z6 - 990 * z4 + 270 * z2 - 10);
    dvalue_ddir.y += SH_C7[9] * sh[58] * 4 * yz * (-143 * z4 + 110 * z2 - 15);
    dvalue_ddir.y += SH_C7[10] * sh[59] * 12 * xy * (-143 * z4 + 66 * z2 - 3);
    dvalue_ddir.y += SH_C7[11] * sh[60] * 8 * yz *
                     (-39 * x2 * z2 + 9 * x2 + 13 * y2 * z2 - 3 * y2);
    dvalue_ddir.y +=
        SH_C7[12] * sh[61] * 40 * xy * (-13 * x2 * z2 + x2 + 13 * y2 * z2 - y2);
    dvalue_ddir.y +=
        SH_C7[13] * sh[62] * 12 * yz * (-5 * x4 + 10 * x2 * y2 - y4);
    dvalue_ddir.y +=
        SH_C7[14] * sh[63] * (-84 * x5 * y + 280 * x3 * y3 - 84 * x * y5);

    dvalue_ddir.z +=
        SH_C7[1] * sh[50] * (12 * x5 * y - 40 * x3 * y3 + 12 * x * y5);
    dvalue_ddir.z += SH_C7[2] * sh[51] * 52 * yz * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.z +=
        SH_C7[3] * sh[52] * 24 * xy * (13 * x2 * z2 - x2 - 13 * y2 * z2 + y2);
    dvalue_ddir.z += SH_C7[4] * sh[53] * 88 * yz *
                     (39 * x2 * z2 - 9 * x2 - 13 * y2 * z2 + 3 * y2);
    dvalue_ddir.z += SH_C7[5] * sh[54] * 20 * xy * (143 * z4 - 66 * z2 + 3);
    dvalue_ddir.z += SH_C7[6] * sh[55] * 36 * yz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.z +=
        SH_C7[7] * sh[56] * (3003 * z6 - 3465 * z4 + 945 * z2 - 35);
    dvalue_ddir.z += SH_C7[8] * sh[57] * 36 * xz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.z += SH_C7[9] * sh[58] *
                     (1430 * x2 * z4 - 660 * x2 * z2 + 30 * x2 -
                      1430 * y2 * z4 + 660 * y2 * z2 - 30 * y2);
    dvalue_ddir.z += SH_C7[10] * sh[59] * 88 * xz *
                     (13 * x2 * z2 - 3 * x2 - 39 * y2 * z2 + 9 * y2);
    dvalue_ddir.z += SH_C7[11] * sh[60] *
                     (78 * x4 * z2 - 6 * x4 - 468 * x2 * y2 * z2 +
                      36 * x2 * y2 + 78 * y4 * z2 - 6 * y4);
    dvalue_ddir.z +=
        SH_C7[12] * sh[61] * 52 * xz * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.z +=
        SH_C7[13] * sh[62] * (2 * x6 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y6);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH8Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;

    dL_dsh[64] = SH_C8[0] * 16 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dL_dsh[65] = SH_C8[1] * 2 * yz * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dL_dsh[66] = SH_C8[2] * 4 * xy*(45 * x4 * z2 - 3 * x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 + 45 * y4 * z2 - 3 * y4);
    dL_dsh[67] = SH_C8[3] * 2 * yz * (25 * x4 * z2 - 5 * x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 + 5 * y4 * z2 - y4);
    dL_dsh[68] = SH_C8[4] *8 * xy * (65 * x2 * z4 - 26 * x2 * z2 + x2 - 65 * y2 * z4 + 26 * y2 * z2 - y2);
    dL_dsh[69] = SH_C8[5] *2 * yz * (117 * x2 * z4 - 78 * x2 * z2 + 9 * x2 - 39 * y2 * z4 + 26 * y2 * z2 - 3 * y2);
    dL_dsh[70] = SH_C8[6] *4 * xy * (143 * z6 - 143 * z4 + 33 * z2 - 1);  
    dL_dsh[71] = SH_C8[7] *2 * yz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    dL_dsh[72] = SH_C8[8] *(6435 * z4 * z4 - 12012 * z6 + 6930 * z4 - 1260 * z2 + 35);
    dL_dsh[73] = SH_C8[9] * 2 * xz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    dL_dsh[74] = SH_C8[10] *(286 * x2 * z6 - 286 * x2 * z4 + 66 * x2 * z2 - 2 * x2 - 286 * y2 * z6 + 286 * y2 * z4 - 66 * y2*z2 + 2*y2);
    dL_dsh[75] = SH_C8[11] * 2* xz*(39*x2*z4 - 26*x2*z2 + 3*x2 - 117*y2*z4 + 78*y2*z2 - 9*y2);
    dL_dsh[76] = SH_C8[12] * (130*x4*z4 - 52*x4*z2 + 2*x4 - 780*x2*y2*z4 + 312*x2*y2*z2 - 12*x2*y2 + 130*y4*z4 - 52*y4*z2 + 2*y4);
    dL_dsh[77] = SH_C8[13] * 2 * xz * (5*x4*z2 - x4 - 50*x2*y2*z2 + 10*x2*y2 + 25*y4*z2 - 5*y4);
    dL_dsh[78] = SH_C8[14] * (30*x6*z2 - 2*x6 - 450*x4*y2*z2 + 30*x4*y2 + 450*x2*y4*z2 - 30*x2*y4 - 30*y6*z2 + 2*y6);
    dL_dsh[79] = SH_C8[15] * 2*xz*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6);
    dL_dsh[80] = SH_C8[16] * (2* x4 * x4 - 56*x6*y2 + 140*x4*y4 - 56*x2*y6 + 2*y4 * y4);

    float3 dvalue_ddir = {0, 0, 0};

    dvalue_ddir.x += SH_C8[0] * sh[64] * 16 * y * (7 * x6 - 35 * x4 * y2 + 21*x2*y4 - y6);
    dvalue_ddir.x += SH_C8[1] * sh[65] * 28 * xyz * (3*x4 - 10*x2*y2 + 3*y4);
    dvalue_ddir.x += SH_C8[2] * sh[66] * 12 *y*(75*x4*z2 - 5*x4 - 150*x2*y2*z2 + 10*x2*y2 + 15*y4*z2 - y4);
    dvalue_ddir.x += SH_C8[3] * sh[67] * 40*xyz*(5*x2*z2 - x2 - 5*y2*z2 + y2);
    dvalue_ddir.x += SH_C8[4] * sh[68] * 8*y*(195*x2*z4 - 78*x2*z2 + 3*x2 - 65*y2*z4 + 26*y2*z2 - y2);
    dvalue_ddir.x += SH_C8[5] * sh[69] * 12*xyz*(39*z4 - 26*z2 + 3);
    dvalue_ddir.x += SH_C8[6] * sh[70] * 4*y*(143*z6 - 143*z4 + 33*z2 - 1);
    dvalue_ddir.x += SH_C8[9] * sh[73] * (1430*z4 * z3 - 2002*z3 * z2 + 770*z3 - 70*z);
    dvalue_ddir.x += SH_C8[10] * sh[74] * 4*x*(143*z6 - 143*z4 + 33*z2 - 1);
    dvalue_ddir.x += SH_C8[11] * sh[75] * 6*z*(39*x2*z4 - 26*x2*z2 + 3*x2 - 39*y2*z4 + 26*y2*z2 - 3*y2);
    dvalue_ddir.x += SH_C8[12] * sh[76] * 8*x*(65*x2*z4 - 26*x2*z2 + x2 - 195*y2*z4 + 78*y2*z2 - 3*y2);
    dvalue_ddir.x += SH_C8[13] * sh[77] * 10*z*(5*x4*z2 - x4 - 30*x2*y2*z2 + 6*x2*y2 + 5*y4*z2 - y4);
    dvalue_ddir.x += SH_C8[14] * sh[78] * 12*x*(15*x4*z2 - x4 - 150*x2*y2*z2 + 10*x2*y2 + 75*y4*z2 - 5*y4);
    dvalue_ddir.x += SH_C8[15] * sh[79] * 14*z*(x6 - 15*x4*y2 + 15*x2*y4 - y6);
    dvalue_ddir.x += SH_C8[16] * sh[80] * 16*x*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6);
    
    dvalue_ddir.y += SH_C8[0] * sh[64] * 16*x*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6);
    dvalue_ddir.y += SH_C8[1] * sh[65] * 14*z*(x6 - 15*x4*y2 + 15*x2*y4 - y6);
    dvalue_ddir.y += SH_C8[2] * sh[66] * 12*x*(15*x4*z2 - x4 - 150*x2*y2*z2 + 10*x2*y2 + 75*y4*z2 - 5*y4);
    dvalue_ddir.y += SH_C8[3] * sh[67] * 10*z*(5*x4*z2 - x4 - 30*x2*y2*z2 + 6*x2*y2 + 5*y4*z2 - y4);
    dvalue_ddir.y += SH_C8[4] * sh[68] * 8*x*(65*x2*z4 - 26*x2*z2 + x2 - 195*y2*z4 + 78*y2*z2 - 3*y2);
    dvalue_ddir.y += SH_C8[5] * sh[69] * 6*z*(39*x2*z4 - 26*x2*z2 + 3*x2 - 39*y2*z4 + 26*y2*z2 - 3*y2);
    dvalue_ddir.y += SH_C8[6] * sh[70] * 4*x*(143*z6 - 143*z4 + 33*z2 - 1);
    dvalue_ddir.y += SH_C8[7] * sh[71] * (1430*z4 * z3 - 2002*z3 * z2 + 770*z3 - 70*z);
    dvalue_ddir.y += SH_C8[10] * sh[74] * 4*y*(-143*z6 + 143*z4 - 33*z2 + 1);
    dvalue_ddir.y += SH_C8[11] * sh[75] * 12*xyz*(-39*z4 + 26*z2 - 3);
    dvalue_ddir.y += SH_C8[12] * sh[76] * 8*y*(-195*x2*z4 + 78*x2*z2 - 3*x2 + 65*y2*z4 - 26*y2*z2 + y2);
    dvalue_ddir.y += SH_C8[13] * sh[77] * 40*xyz*(-5*x2*z2 + x2 + 5*y2*z2 - y2);
    dvalue_ddir.y += SH_C8[14] * sh[78] * 12*y*(-75*x4*z2 + 5*x4 + 150*x2*y2*z2 - 10*x2*y2 - 15*y4*z2 + y4);
    dvalue_ddir.y += SH_C8[15] * sh[79] * 28*xyz*(-3*x4 + 10*x2*y2 - 3*y4);
    dvalue_ddir.y += SH_C8[16] * sh[80] * 16*y*(-7*x6 + 35*x4*y2 - 21*x2*y4 + y6);

    dvalue_ddir.z += SH_C8[1] * sh[65] * 2*y*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
    dvalue_ddir.z += SH_C8[2] * sh[66] * 120*xyz*(3*x4 - 10*x2*y2 + 3*y4);
    dvalue_ddir.z += SH_C8[3] * sh[67] * 2*y*(75*x4*z2 - 5*x4 - 150*x2*y2*z2 + 10*x2*y2 + 15*y4*z2 - y4);
    dvalue_ddir.z += SH_C8[4] * sh[68] * 416*xyz*(5*x2*z2 - x2 - 5*y2*z2 + y2);
    dvalue_ddir.z += SH_C8[5] * sh[69] * 6*y*(195*x2*z4 - 78*x2*z2 + 3*x2 - 65*y2*z4 + 26*y2*z2 - y2);
    dvalue_ddir.z += SH_C8[6] * sh[70] * 88*xyz*(39*z4 - 26*z2 + 3);
    dvalue_ddir.z += SH_C8[7] * sh[71] * 70*y*(143*z6 - 143*z4 + 33*z2 - 1);
    dvalue_ddir.z += SH_C8[8] * sh[72] * (51480*z4 * z3 - 72072*z3 * z2 + 27720*z3 - 2520*z);
    dvalue_ddir.z += SH_C8[9] * sh[73] * 70*x*(143*z6 - 143*z4 + 33*z2 - 1);
    dvalue_ddir.z += SH_C8[10] * sh[74] * 44*z*(39*x2*z4 - 26*x2*z2 + 3*x2 - 39*y2*z4 + 26*y2*z2 - 3*y2);
    dvalue_ddir.z += SH_C8[11] * sh[75] * 6*x*(65*x2*z4 - 26*x2*z2 + x2 - 195*y2*z4 + 78*y2*z2 - 3*y2);
    dvalue_ddir.z += SH_C8[12] * sh[76] * 104*z*(5*x4*z2 - x4 - 30*x2*y2*z2 + 6*x2*y2 + 5*y4*z2 - y4);
    dvalue_ddir.z += SH_C8[13] * sh[77] * 2*x*(15*x4*z2 - x4 - 150*x2*y2*z2 + 10*x2*y2 + 75*y4*z2 - 5*y4);
    dvalue_ddir.z += SH_C8[14] * sh[78] * 60*z*(x6 - 15*x4*y2 + 15*x2*y4 - y6);
    dvalue_ddir.z += SH_C8[15] * sh[79] *  2*x*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6);

    dL_ddir[0].x += dvalue_ddir.x * dL_dvalue;
    dL_ddir[0].y += dvalue_ddir.y * dL_dvalue;
    dL_ddir[0].z += dvalue_ddir.z * dL_dvalue;
}

__forceinline__ __device__ void evaluateSH9Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    
}

__forceinline__ __device__ void evaluateSH10Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dvalue,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    
}

__global__ void computeSHForwardCUDAKernel(const int P,
                                           const int C,
                                           const int D,
                                           const float *shs,
                                           const float3 *dirs,
                                           const bool *visible,
                                           float *value) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visible[idx])
        return;

    for (int i = 0; i < C; i++) {
        float *value_tmp = value + C * idx;
        const float *sh_tmp = shs + idx * D * C + i * D;

        value_tmp[i] = evaluateSH0Forward(sh_tmp);
        if (D > 1)
            value_tmp[i] += evaluateSH1Forward(sh_tmp, dirs[idx]);
        if (D > 4)
            value_tmp[i] += evaluateSH2Forward(sh_tmp, dirs[idx]);
        if (D > 9)
            value_tmp[i] += evaluateSH3Forward(sh_tmp, dirs[idx]);
        if (D > 16)
            value_tmp[i] += evaluateSH4Forward(sh_tmp, dirs[idx]);
        if (D > 25)
            value_tmp[i] += evaluateSH5Forward(sh_tmp, dirs[idx]);
        if (D > 36)
            value_tmp[i] += evaluateSH6Forward(sh_tmp, dirs[idx]);
        if (D > 49)
            value_tmp[i] += evaluateSH7Forward(sh_tmp, dirs[idx]);
        if (D > 64)
            value_tmp[i] += evaluateSH8Forward(sh_tmp, dirs[idx]);
        if (D > 91)
            value_tmp[i] += evaluateSH9Forward(sh_tmp, dirs[idx]);
        if (D > 100)
            value_tmp[i] += evaluateSH10Forward(sh_tmp, dirs[idx]);
    }
}

__global__ void computeSHBackwardCUDAKernel(const int P,
                                            const int C,
                                            const int D,
                                            const float *shs,
                                            const float3 *dirs,
                                            const bool *visible,
                                            const float *dL_dvalue,
                                            float *dL_dshs,
                                            float3 *dL_ddirs) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visible[idx])
        return;

    // SH: [N, C, D]
    // dir: [N, 3]
    // dL_dvalue: [N, C]
    for (int i = 0; i < C; i++) {
        const float *dL_dvalue_tmp = dL_dvalue + idx * C;
        const float *sh_tmp = shs + idx * D * C + i * D;
        float *dL_dsh_tmp = dL_dshs + idx * D * C + i * D;
        float3 *dL_ddir_tmp = dL_ddirs + idx;

        evaluateSH0Backward(
            sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 1)
            evaluateSH1Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 4)
            evaluateSH2Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 9)
            evaluateSH3Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 16)
            evaluateSH4Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 25)
            evaluateSH5Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 36)
            evaluateSH6Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 49)
            evaluateSH7Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 64)
            evaluateSH8Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 81)
            evaluateSH9Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 100)
            evaluateSH10Backward(
                sh_tmp, dirs[idx], dL_dvalue_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
    }
}

torch::Tensor computeSHForward(const torch::Tensor &shs,
                               const torch::Tensor &view_dirs,
                               const torch::Tensor &visible) {
    CHECK_INPUT(shs);
    CHECK_INPUT(view_dirs);
    CHECK_INPUT(visible);

    const int P = shs.size(0);
    const int C = shs.size(1);
    const int D = shs.size(2);
    auto float_opts = shs.options().dtype(torch::kFloat32);
    torch::Tensor value = torch::zeros({P, C}, float_opts);

    if (P != 0) {
        computeSHForwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            C,
            D,
            shs.contiguous().data_ptr<float>(),
            (const float3 *)view_dirs.contiguous().data_ptr<float>(),
            visible.contiguous().data_ptr<bool>(),
            value.data_ptr<float>());
    }

    return value;
}

std::tuple<torch::Tensor, torch::Tensor>
computeSHBackward(const torch::Tensor &shs,
                  const torch::Tensor &view_dirs,
                  const torch::Tensor &visible,
                  const torch::Tensor &dL_dvalue) {
    CHECK_INPUT(shs);
    CHECK_INPUT(view_dirs);
    CHECK_INPUT(visible);
    CHECK_INPUT(dL_dvalue);

    const int P = shs.size(0);
    const int C = shs.size(1);
    const int D = shs.size(2);
    auto float_opts = shs.options().dtype(torch::kFloat32);
    torch::Tensor dL_dshs = torch::zeros({P, C, D}, float_opts);
    torch::Tensor dL_dvdirs = torch::zeros({P, 3}, float_opts);

    if (P != 0) {
        computeSHBackwardCUDAKernel<<<(P + 255) / 256, 256>>>(
            P,
            C,
            D,
            shs.contiguous().data_ptr<float>(),
            (const float3 *)view_dirs.contiguous().data_ptr<float>(),
            visible.contiguous().data_ptr<bool>(),
            dL_dvalue.contiguous().data_ptr<float>(),
            dL_dshs.data_ptr<float>(),
            (float3 *)dL_dvdirs.data_ptr<float>());
    }

    return std::make_tuple(dL_dshs, dL_dvdirs);
}