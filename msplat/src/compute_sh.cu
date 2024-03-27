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

// clang-format off

__device__ const float SH_C0 = 0.28209479177387814f;

__device__ const float SH_C1[] = {
    -0.4886025119029199f, 
    0.4886025119029199f
};

__device__ const float SH_C2[] = {
    1.0925484305920792f, 
    -1.0925484305920792f, 
    0.31539156525252005f
};

__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f
};

__device__ const float SH_C4[] = {
    0.6258357354491761f,
    -1.7701307697799304f,
    0.47308734787878004f,
    -0.6690465435572892f,
    0.10578554691520431f
};

__device__ const float SH_C5[] = {
    -0.3281910284200851f,
    1.0378311574405208f,
    -0.2446191497176252f,
    1.198384196243331f,
    -0.22647332559784847f,
    0.1169503224534236f
};

__device__ const float SH_C6[] = {
    0.34159205259595715f,
    -1.183309581115876f,
    0.2522824503643621f,
    -0.46060262975746175f,
    0.23030131487873087f,
    -0.2913106812593657f,
    0.06356920226762842f
};

__device__ const float SH_C7[] = {
    -0.3535813662622981f,
    1.32298033090095f,
    -0.2594577893601302f,
    0.5189155787202604f,
    -0.07822946693114702f,
    0.11063317311124565f,
    -0.04516580379125865f,
    0.06828427691200495f
};

__device__ const float SH_C8[] = {
    0.36446333008741494f,
    -1.4578533203496598f,
    0.2661663830297713f,
    -1.724955311049054f,
    0.23920826237966533f,
    -0.6176330776477721f,
    0.22807612921745474f,
    -0.054520622949389974f,
    0.009086770491564996f
};

__device__ const float SH_C9[] = {
    -0.37445047592659414f,
    1.5886588244773487f,
    -0.2724527406720266f,
    0.6292026526737834f,
    -0.2436891395195093f,
    2.0388496193645866f,
    -0.23085426000809722f,
    0.30225917742574926f,
    -0.03222093657611365f,
    0.009606427264386591f
};

__device__ const float SH_C10[] = {
    0.383697559110995f,
    -1.7159476499458572f,
    0.2783634663602092f,
    -0.6818484556149027f,
    0.08268627068130938f,
    -0.1479136976394845f,
    0.23387209083912108f,
    -0.3307450827252375f,
    0.032432223670016275f,
    -0.037449506132604095f,
    0.005049690376783604f,
};

// clang-format on

__forceinline__ __device__ float evaluateSH0Forward(const float *sh) {
    return SH_C0 * sh[0];
}

__forceinline__ __device__ float evaluateSH1Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;

    float result = 0;
    result += sh[1] * SH_C1[0] * y;
    result += sh[2] * SH_C1[1] * z;
    result += sh[3] * SH_C1[0] * x;

    return result;
}

__forceinline__ __device__ float evaluateSH2Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = 0;
    result += sh[4] * SH_C2[0] * xy;
    result += sh[5] * SH_C2[1] * yz;
    result += sh[6] * SH_C2[2] * (2.0f * z2 - x2 - y2);
    result += sh[7] * SH_C2[1] * xz;
    result += sh[8] * SH_C2[0] * 0.5 * (x2 - y2);

    return result;
}

__forceinline__ __device__ float evaluateSH3Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = 0;
    result += sh[9] * SH_C3[0] * y * (3.0f * x2 - y2);
    result += sh[10] * SH_C3[1] * xy * z;
    result += sh[11] * SH_C3[2] * y * (4.0f * z2 - x2 - y2);
    result += sh[12] * SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2);
    result += sh[13] * SH_C3[2] * x * (4.0f * z2 - x2 - y2);
    result += sh[14] * SH_C3[1] * z * 0.5 * (x2 - y2);
    result += sh[15] * SH_C3[0] * x * (x2 - 3.0f * y2);

    return result;
}

__forceinline__ __device__ float evaluateSH4Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float result = 0;
    result += sh[16] * SH_C4[0] * 4 * xy * (x2 - y2);
    result += sh[17] * SH_C4[1] * yz * (3 * x2 - y2);
    result += sh[18] * SH_C4[2] * 2 * xy * (7 * z2 - 1);
    result += sh[19] * SH_C4[3] * yz * (7 * z2 - 3);
    result += sh[20] * SH_C4[4] * (z2 * (35 * z2 - 30) + 3);
    result += sh[21] * SH_C4[3] * xz * (7 * z2 - 3);
    result += sh[22] * SH_C4[2] * (x2 - y2) * (7 * z2 - 1);
    result += sh[23] * SH_C4[1] * xz * (x2 - 3 * y2);
    result += sh[24] * SH_C4[0] * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2));

    return result;
}

__forceinline__ __device__ float evaluateSH5Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    float result = 0;
    result += sh[25] * SH_C5[0] * (10 * x3 * xy - 20 * x2 * y3 + 2 * y3 * y2);
    result += sh[26] * SH_C5[1] * (8 * x3 * yz - 8 * xy * y2 * z);
    result += sh[27] * SH_C5[2] *
              (54 * x2 * y * z2 - 6 * x2 * y - 18 * y3 * z2 + 2 * y3);
    result += sh[28] * SH_C5[3] * (12 * xy * z3 - 4 * x * yz);
    result += sh[29] * SH_C5[4] * (42 * yz * z3 - 28 * y * z2 + 2 * y);
    result += sh[30] * SH_C5[5] * (63 * z2 * z3 - 70 * z3 + 15 * z);
    result += sh[31] * SH_C5[4] * (42 * x * z2 * z2 - 28 * x * z2 + 2 * x);
    result += sh[32] * SH_C5[3] *
              (6 * x2 * z3 - 2 * x2 * z - 6 * y2 * z3 + 2 * y2 * z);
    result += sh[33] * SH_C5[2] *
              (18 * x3 * z2 - 2 * x3 - 54 * x * y2 * z2 + 6 * x * y2);
    result += sh[34] * SH_C5[1] *
              (2 * x2 * x2 * z - 12 * x2 * y2 * z + 2 * y2 * y2 * z);
    result += sh[35] * SH_C5[0] * (2 * x3 * x2 - 20 * x3 * y2 + 10 * xy * y3);

    return result;
}

__forceinline__ __device__ float evaluateSH6Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    float result = 0;
    result +=
        sh[36] * SH_C6[0] * (12 * x3 * x2 * y - 40 * x3 * y3 + 12 * xy * y4);
    result +=
        sh[37] * SH_C6[1] * (10 * x4 * yz - 20 * x2 * y2 * yz + 2 * y4 * yz);
    result +=
        sh[38] * SH_C6[2] *
        (88 * x2 * xy * z2 - 8 * x2 * xy - 88 * xy * y2 * z2 + 8 * xy * y2);
    result += sh[39] * SH_C6[3] * 2 * yz *
              (33 * x2 * z2 - 9 * x2 - 11 * y2 * z2 + 3 * y2);
    result += sh[40] * SH_C6[4] * 4 * xy * (33 * z4 - 18 * z2 + 1);
    result += sh[41] * SH_C6[5] * 2 * yz * (33 * z4 - 30 * z2 + 5);
    result += sh[42] * SH_C6[6] * (231 * z3 * z3 - 315 * z4 + 105 * z2 - 5);
    result += sh[43] * SH_C6[5] * (66 * xz * z4 - 60 * xz * z2 + 10 * xz);
    result += sh[44] * SH_C6[4] *
              (66 * x2 * z4 - 36 * x2 * z2 + 2 * x2 - 66 * y2 * z4 +
               36 * y2 * z2 - 2 * y2);
    result += sh[45] * SH_C6[3] *
              (22 * x3 * z3 - 6 * x2 * xz - 66 * xy * yz * z2 + 18 * xy * yz);
    result += sh[46] * SH_C6[2] *
              (22 * x4 * z2 - 2 * x4 - 132 * x2 * y2 * z2 + 12 * x2 * y2 +
               22 * y4 * z2 - 2 * y4);
    result += sh[47] * SH_C6[1] *
              (2 * x4 * xz - 20 * x2 * xy * yz + 10 * xy * y2 * yz);
    result += sh[48] * SH_C6[0] *
              (2 * x4 * x2 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y3 * y3);

    return result;
}

__forceinline__ __device__ float evaluateSH7Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    float result = 0;
    result += sh[49] * SH_C7[0] * 2 * y *
              (7 * x3 * x3 - 35 * x4 * y2 + 21 * x2 * y4 - y3 * y3);
    result += sh[50] * SH_C7[1] * 4 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4);
    result += sh[51] * SH_C7[2] * 2 * y *
              (65 * x4 * z2 - 5 * x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
               13 * y4 * z2 - y4);
    result += sh[52] * SH_C7[3] * 8 * xyz *
              (13 * x2 * z2 - 3 * x2 - 13 * y2 * z2 + 3 * y2);
    result += sh[53] * SH_C7[4] * 2 * y *
              (429 * x2 * z4 - 198 * x2 * z2 + 9 * x2 - 143 * y2 * z4 +
               66 * y2 * z2 - 3 * y2);
    result += sh[54] * SH_C7[5] * 4 * xyz * (143 * z4 - 110 * z2 + 15);
    result +=
        sh[55] * SH_C7[6] * 2 * y * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    result +=
        sh[56] * SH_C7[7] * z * (429 * z3 * z3 - 693 * z4 + 315 * z2 - 35);
    result +=
        sh[57] * SH_C7[6] * 2 * x * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    result += sh[58] * SH_C7[5] * 2 * z *
              (143 * x2 * z4 - 110 * x2 * z2 + 15 * x2 - 143 * y2 * z4 +
               110 * y2 * z2 - 15 * y2);
    result += sh[59] * SH_C7[4] * 2 * x *
              (143 * x2 * z4 - 66 * x2 * z2 + 3 * x2 - 429 * y2 * z4 +
               198 * y2 * z2 - 9 * y2);
    result += sh[60] * SH_C7[3] * 2 * z *
              (13 * x4 * z2 - 3 * x4 - 78 * x2 * y2 * z2 + 18 * x2 * y2 +
               13 * y4 * z2 - 3 * y4);
    result += sh[61] * SH_C7[2] * 2 * x *
              (13 * x4 * z2 - x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
               65 * y4 * z2 - 5 * y4);
    result += sh[62] * SH_C7[1] * 2 * z *
              (x3 * x3 - 15 * x4 * y2 + 15 * x2 * y4 - y3 * y3);
    result += sh[63] * SH_C7[0] * 2 * x *
              (x3 * x3 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y3 * y3);

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

    float result = 0;
    result +=
        sh[64] * SH_C8[0] * 16 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    result += sh[65] * SH_C8[1] * 2 * yz *
              (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    result += sh[66] * SH_C8[2] * 4 * xy *
              (45 * x4 * z2 - 3 * x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 +
               45 * y4 * z2 - 3 * y4);
    result += sh[67] * SH_C8[3] * 2 * yz *
              (25 * x4 * z2 - 5 * x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 +
               5 * y4 * z2 - y4);
    result +=
        sh[68] * SH_C8[4] * 8 * xy *
        (65 * x2 * z4 - 26 * x2 * z2 + x2 - 65 * y2 * z4 + 26 * y2 * z2 - y2);
    result += sh[69] * SH_C8[5] * 2 * yz *
              (117 * x2 * z4 - 78 * x2 * z2 + 9 * x2 - 39 * y2 * z4 +
               26 * y2 * z2 - 3 * y2);
    result += sh[70] * SH_C8[6] * 4 * xy * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    result +=
        sh[71] * SH_C8[7] * 2 * yz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    result += sh[72] * SH_C8[8] *
              (6435 * z4 * z4 - 12012 * z6 + 6930 * z4 - 1260 * z2 + 35);
    result +=
        sh[73] * SH_C8[7] * 2 * xz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    result += sh[74] * SH_C8[6] *
              (286 * x2 * z6 - 286 * x2 * z4 + 66 * x2 * z2 - 2 * x2 -
               286 * y2 * z6 + 286 * y2 * z4 - 66 * y2 * z2 + 2 * y2);
    result += sh[75] * SH_C8[5] * 2 * xz *
              (39 * x2 * z4 - 26 * x2 * z2 + 3 * x2 - 117 * y2 * z4 +
               78 * y2 * z2 - 9 * y2);
    result += sh[76] * SH_C8[4] *
              (130 * x4 * z4 - 52 * x4 * z2 + 2 * x4 - 780 * x2 * y2 * z4 +
               312 * x2 * y2 * z2 - 12 * x2 * y2 + 130 * y4 * z4 -
               52 * y4 * z2 + 2 * y4);
    result += sh[77] * SH_C8[3] * 2 * xz *
              (5 * x4 * z2 - x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 +
               25 * y4 * z2 - 5 * y4);
    result += sh[78] * SH_C8[2] *
              (30 * x6 * z2 - 2 * x6 - 450 * x4 * y2 * z2 + 30 * x4 * y2 +
               450 * x2 * y4 * z2 - 30 * x2 * y4 - 30 * y6 * z2 + 2 * y6);
    result += sh[79] * SH_C8[1] * 2 * xz *
              (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    result += sh[80] * SH_C8[0] *
              (2 * x4 * x4 - 56 * x6 * y2 + 140 * x4 * y4 - 56 * x2 * y6 +
               2 * y4 * y4);

    return result;
}

__forceinline__ __device__ float evaluateSH9Forward(const float *sh,
                                                    const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;
    float x8 = x4 * x4, y8 = y4 * y4, z8 = z4 * z4;

    float result = 0;
    result += sh[81] * SH_C9[0] * 2 * y *
              (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    result +=
        sh[82] * SH_C9[1] * 16 * xyz * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    result += sh[83] * SH_C9[2] * 2 * y *
              (119 * x6 * z2 - 7 * x6 - 595 * x4 * y2 * z2 + 35 * x4 * y2 +
               357 * x2 * y4 * z2 - 21 * x2 * y4 - 17 * y6 * z2 + y6);
    result += sh[84] * SH_C9[3] * 4 * xyz *
              (51 * x4 * z2 - 9 * x4 - 170 * x2 * y2 * z2 + 30 * x2 * y2 +
               51 * y4 * z2 - 9 * y4);
    result +=
        sh[85] * SH_C9[4] * 2 * y *
        (425 * x4 * z4 - 150 * x4 * z2 + 5 * x4 - 850 * x2 * y2 * z4 +
         300 * x2 * y2 * z2 - 10 * x2 * y2 + 85 * y4 * z4 - 30 * y4 * z2 + y4);
    result +=
        sh[86] * SH_C9[5] * 8 * xyz *
        (17 * x2 * z4 - 10 * x2 * z2 + x2 - 17 * y2 * z4 + 10 * y2 * z2 - y2);
    result += sh[87] * SH_C9[6] * 2 * y *
              (663 * x2 * z6 - 585 * x2 * z4 + 117 * x2 * z2 - 3 * x2 -
               221 * y2 * z6 + 195 * y2 * z4 - 39 * y2 * z2 + y2);
    result += sh[88] * SH_C9[7] * 4 * xyz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    result += sh[89] * SH_C9[8] * 2 * y *
              (2431 * z8 - 4004 * z6 + 2002 * z4 - 308 * z2 + 7);
    result += sh[90] * SH_C9[9] * z *
              (12155 * z8 - 25740 * z6 + 18018 * z4 - 4620 * z2 + 315);
    result += sh[91] * SH_C9[8] * 2 * x *
              (2431 * z8 - 4004 * z6 + 2002 * z4 - 308 * z2 + 7);
    result += sh[92] * SH_C9[7] * 2 * z *
              (221 * x2 * z6 - 273 * x2 * z4 + 91 * x2 * z2 - 7 * x2 -
               221 * y2 * z6 + 273 * y2 * z4 - 91 * y2 * z2 + 7 * y2);
    result += sh[93] * SH_C9[6] * 2 * x *
              (221 * x2 * z6 - 195 * x2 * z4 + 39 * x2 * z2 - x2 -
               663 * y2 * z6 + 585 * y2 * z4 - 117 * y2 * z2 + 3 * y2);
    result +=
        sh[94] * SH_C9[5] * 2 * z *
        (17 * x4 * z4 - 10 * x4 * z2 + x4 - 102 * x2 * y2 * z4 +
         60 * x2 * y2 * z2 - 6 * x2 * y2 + 17 * y4 * z4 - 10 * y4 * z2 + y4);
    result += sh[95] * SH_C9[4] * 2 * x *
              (85 * x4 * z4 - 30 * x4 * z2 + x4 - 850 * x2 * y2 * z4 +
               300 * x2 * y2 * z2 - 10 * x2 * y2 + 425 * y4 * z4 -
               150 * y4 * z2 + 5 * y4);
    result += sh[96] * SH_C9[3] * 2 * z *
              (17 * x6 * z2 - 3 * x6 - 255 * x4 * y2 * z2 + 45 * x4 * y2 +
               255 * x2 * y4 * z2 - 45 * x2 * y4 - 17 * y6 * z2 + 3 * y6);
    result += sh[97] * SH_C9[2] * 2 * x *
              (17 * x6 * z2 - x6 - 357 * x4 * y2 * z2 + 21 * x4 * y2 +
               595 * x2 * y4 * z2 - 35 * x2 * y4 - 119 * y6 * z2 + 7 * y6);
    result += sh[98] * SH_C9[1] * 2 * z *
              (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8);
    result += sh[99] * SH_C9[0] * 2 * x *
              (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);

    return result;
}

__forceinline__ __device__ float evaluateSH10Forward(const float *sh,
                                                     const float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;
    float x8 = x4 * x4, y8 = y4 * y4, z8 = z4 * z4;
    float x10 = x6 * x4, y10 = y6 * y4, z10 = z6 * z4;

    float result = 0;
    result +=
        sh[100] * SH_C10[0] * xy *
        (20 * x8 - 240 * x6 * y2 + 504 * x4 * y4 - 240 * x2 * y6 + 20 * y8);
    result += sh[101] * SH_C10[1] * 2 * yz *
              (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    result += sh[102] * SH_C10[2] * 16 * xy *
              (19 * x6 * z2 - x6 - 133 * x4 * y2 * z2 + 7 * x4 * y2 +
               133 * x2 * y4 * z2 - 7 * x2 * y4 - 19 * y6 * z2 + y6);
    result += sh[103] * SH_C10[3] * 2 * yz *
              (133 * x6 * z2 - 21 * x6 - 665 * x4 * y2 * z2 + 105 * x4 * y2 +
               399 * x2 * y4 * z2 - 63 * x2 * y4 - 19 * y6 * z2 + 3 * y6);
    result += sh[104] * SH_C10[4] * 4 * xy *
              (969 * x4 * z4 - 306 * x4 * z2 + 9 * x4 - 3230 * x2 * y2 * z4 +
               1020 * x2 * y2 * z2 - 30 * x2 * y2 + 969 * y4 * z4 -
               306 * y4 * z2 + 9 * y4);
    result += sh[105] * SH_C10[5] * 2 * yz *
              (1615 * x4 * z4 - 850 * x4 * z2 + 75 * x4 - 3230 * x2 * y2 * z4 +
               1700 * x2 * y2 * z2 - 150 * x2 * y2 + 323 * y4 * z4 -
               170 * y4 * z2 + 15 * y4);
    result += sh[106] * SH_C10[6] * 8 * xy *
              (323 * x2 * z6 - 255 * x2 * z4 + 45 * x2 * z2 - x2 -
               323 * y2 * z6 + 255 * y2 * z4 - 45 * y2 * z2 + y2);
    result += sh[107] * SH_C10[7] * 2 * yz *
              (969 * x2 * z6 - 1071 * x2 * z4 + 315 * x2 * z2 - 21 * x2 -
               323 * y2 * z6 + 357 * y2 * z4 - 105 * y2 * z2 + 7 * y2);
    result += sh[108] * SH_C10[8] * 4 * xy *
              (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    result += sh[109] * SH_C10[9] * 2 * yz *
              (4199 * z8 - 7956 * z6 + 4914 * z4 - 1092 * z2 + 63);
    result +=
        sh[110] * SH_C10[10] *
        (46189 * z10 - 109395 * z8 + 90090 * z6 - 30030 * z4 + 3465 * z2 - 63);
    result += sh[111] * SH_C10[9] * 2 * xz *
              (4199 * z8 - 7956 * z6 + 4914 * z4 - 1092 * z2 + 63);
    result += sh[112] * SH_C10[8] *
              (8398 * x2 * z8 - 12376 * x2 * z6 + 5460 * x2 * z4 -
               728 * x2 * z2 + 14 * x2 - 8398 * y2 * z8 + 12376 * y2 * z6 -
               5460 * y2 * z4 + 728 * y2 * z2 - 14 * y2);
    result += sh[113] * SH_C10[7] * 2 * xz *
              (323 * x2 * z6 - 357 * x2 * z4 + 105 * x2 * z2 - 7 * x2 -
               969 * y2 * z6 + 1071 * y2 * z4 - 315 * y2 * z2 + 21 * y2);
    result +=
        sh[114] * SH_C10[6] *
        (646 * x4 * z6 - 510 * x4 * z4 + 90 * x4 * z2 - 2 * x4 -
         3876 * x2 * y2 * z6 + 3060 * x2 * y2 * z4 - 540 * x2 * y2 * z2 +
         12 * x2 * y2 + 646 * y4 * z6 - 510 * y4 * z4 + 90 * y4 * z2 - 2 * y4);
    result += sh[115] * SH_C10[5] * 2 * xz *
              (323 * x4 * z4 - 170 * x4 * z2 + 15 * x4 - 3230 * x2 * y2 * z4 +
               1700 * x2 * y2 * z2 - 150 * x2 * y2 + 1615 * y4 * z4 -
               850 * y4 * z2 + 75 * y4);
    result += sh[116] * SH_C10[4] *
              (646 * x6 * z4 - 204 * x6 * z2 + 6 * x6 - 9690 * x4 * y2 * z4 +
               3060 * x4 * y2 * z2 - 90 * x4 * y2 + 9690 * x2 * y4 * z4 -
               3060 * x2 * y4 * z2 + 90 * x2 * y4 - 646 * y6 * z4 +
               204 * y6 * z2 - 6 * y6);
    result += sh[117] * SH_C10[3] * 2 * xz *
              (19 * x6 * z2 - 3 * x6 - 399 * x4 * y2 * z2 + 63 * x4 * y2 +
               665 * x2 * y4 * z2 - 105 * x2 * y4 - 133 * y6 * z2 + 21 * y6);
    result += sh[118] * SH_C10[2] *
              (38 * x8 * z2 - 2 * x8 - 1064 * x6 * y2 * z2 + 56 * x6 * y2 +
               2660 * x4 * y4 * z2 - 140 * x4 * y4 - 1064 * x2 * y6 * z2 +
               56 * x2 * y6 + 38 * y8 * z2 - 2 * y8);
    result += sh[119] * SH_C10[1] * 2 * xz *
              (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);
    result += sh[120] * SH_C10[0] *
              (2 * x10 - 90 * x8 * y2 + 420 * x6 * y4 - 420 * x4 * y6 +
               90 * x2 * y8 - 2 * y10);

    return result;
}

__forceinline__ __device__ void evaluateSH0Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    dL_dsh[0] = dL_dval * SH_C0;
}

__forceinline__ __device__ void evaluateSH1Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    dL_dsh[1] = dL_dval * SH_C1[0] * dir.y;
    dL_dsh[2] = dL_dval * SH_C1[1] * dir.z;
    dL_dsh[3] = dL_dval * SH_C1[0] * dir.x;

    float3 dvalue_ddir = {SH_C1[0] * sh[3], SH_C1[0] * sh[1], SH_C1[1] * sh[2]};

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH2Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    dL_dsh[4] = dL_dval * SH_C2[0] * xy;
    dL_dsh[5] = dL_dval * SH_C2[1] * yz;
    dL_dsh[6] = dL_dval * SH_C2[2] * (2.0f * z2 - x2 - y2);
    dL_dsh[7] = dL_dval * SH_C2[1] * xz;
    dL_dsh[8] = dL_dval * SH_C2[0] * 0.5 * (x2 - y2);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C2[0] * sh[4] * y;
    dvalue_ddir.x += SH_C2[2] * sh[6] * (-2 * x);
    dvalue_ddir.x += SH_C2[1] * sh[7] * z;
    dvalue_ddir.x += SH_C2[0] * sh[8] * x;

    dvalue_ddir.y += SH_C2[0] * sh[4] * x;
    dvalue_ddir.y += SH_C2[1] * sh[5] * z;
    dvalue_ddir.y += SH_C2[2] * sh[6] * (-2 * y);
    dvalue_ddir.y += SH_C2[0] * sh[8] * (-y);

    dvalue_ddir.z += SH_C2[1] * sh[5] * y;
    dvalue_ddir.z += SH_C2[2] * sh[6] * 4 * z;
    dvalue_ddir.z += SH_C2[1] * sh[7] * x;

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH3Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    dL_dsh[9] = dL_dval * SH_C3[0] * y * (3.0f * x2 - y2);
    dL_dsh[10] = dL_dval * SH_C3[1] * xy * z;
    dL_dsh[11] = dL_dval * SH_C3[2] * y * (4.0f * z2 - x2 - y2);
    dL_dsh[12] = dL_dval * SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2);
    dL_dsh[13] = dL_dval * SH_C3[2] * x * (4.0f * z2 - x2 - y2);
    dL_dsh[14] = dL_dval * SH_C3[1] * z * 0.5 * (x2 - y2);
    dL_dsh[15] = dL_dval * SH_C3[0] * x * (x2 - 3.0f * y2);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C3[0] * sh[9] * 6 * xy;
    dvalue_ddir.x += SH_C3[1] * sh[10] * yz;
    dvalue_ddir.x += SH_C3[2] * sh[11] * (-2 * xy);
    dvalue_ddir.x += SH_C3[3] * sh[12] * (-6 * xz);
    dvalue_ddir.x += SH_C3[2] * sh[13] * (4 * z2 - 3 * x2 - y2);
    dvalue_ddir.x += SH_C3[1] * sh[14] * xz;
    dvalue_ddir.x += SH_C3[0] * sh[15] * (3 * x2 - 3 * y2);

    dvalue_ddir.y += SH_C3[0] * sh[9] * (3 * x2 - 3 * y2);
    dvalue_ddir.y += SH_C3[1] * sh[10] * xz;
    dvalue_ddir.y += SH_C3[2] * sh[11] * (4 * z2 - x2 - 3 * y2);
    dvalue_ddir.y += SH_C3[3] * sh[12] * (-6.0 * yz);
    dvalue_ddir.y += SH_C3[2] * sh[13] * (-2.0 * xy);
    dvalue_ddir.y += SH_C3[1] * sh[14] * (-yz);
    dvalue_ddir.y += SH_C3[0] * sh[15] * (-6.0 * xy);

    dvalue_ddir.z += SH_C3[1] * sh[10] * xy;
    dvalue_ddir.z += SH_C3[2] * sh[11] * 8 * yz;
    dvalue_ddir.z += SH_C3[3] * sh[12] * (6 * z2 - 3 * x2 - 3 * y2);
    dvalue_ddir.z += SH_C3[2] * sh[13] * 8 * xz;
    dvalue_ddir.z += SH_C3[1] * sh[14] * 0.5 * (x2 - y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH4Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    dL_dsh[16] = dL_dval * SH_C4[0] * 4 * xy * (x2 - y2);
    dL_dsh[17] = dL_dval * SH_C4[1] * yz * (3 * x2 - y2);
    dL_dsh[18] = dL_dval * SH_C4[2] * 2 * xy * (7 * z2 - 1);
    dL_dsh[19] = dL_dval * SH_C4[3] * yz * (7 * z2 - 3);
    dL_dsh[20] = dL_dval * SH_C4[4] * (z2 * (35 * z2 - 30) + 3);
    dL_dsh[21] = dL_dval * SH_C4[3] * xz * (7 * z2 - 3);
    dL_dsh[22] = dL_dval * SH_C4[2] * (x2 - y2) * (7 * z2 - 1);
    dL_dsh[23] = dL_dval * SH_C4[1] * xz * (x2 - 3 * y2);
    dL_dsh[24] = dL_dval * SH_C4[0] * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2));

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C4[0] * sh[16] * 4 * (3 * x2 * y - y3);
    dvalue_ddir.x += SH_C4[1] * sh[17] * 6 * xyz;
    dvalue_ddir.x += SH_C4[2] * sh[18] * 2 * (7 * y * z2 - y);
    dvalue_ddir.x += SH_C4[3] * sh[21] * (7 * z3 - 3 * z);
    dvalue_ddir.x += SH_C4[2] * sh[22] * (14 * x * z2 - 2 * x);
    dvalue_ddir.x += SH_C4[1] * sh[23] * (3 * x2 * z - 3 * y2 * z);
    dvalue_ddir.x += SH_C4[0] * sh[24] * (4 * x3 - 6 * x * y2 - 6 * x * y2);

    dvalue_ddir.y += SH_C4[0] * sh[16] * 4 * (x3 - 3 * x * y2);
    dvalue_ddir.y += SH_C4[1] * sh[17] * (3 * x2 * z - 3 * y2 * z);
    dvalue_ddir.y += SH_C4[2] * sh[18] * 2 * (7 * x * z2 - x);
    dvalue_ddir.y += SH_C4[3] * sh[19] * (7 * z3 - 3 * z);
    dvalue_ddir.y += SH_C4[2] * sh[22] * (-14 * y * z2 + 2 * y);
    dvalue_ddir.y += SH_C4[1] * sh[23] * (-6 * xyz);
    dvalue_ddir.y += SH_C4[0] * sh[24] * (-12 * x2 * y + 4 * y3);

    dvalue_ddir.z += SH_C4[1] * sh[17] * (3 * x2 * y - y3);
    dvalue_ddir.z += SH_C4[2] * sh[18] * 28 * xyz;
    dvalue_ddir.z += SH_C4[3] * sh[19] * (21 * y * z2 - 3 * y);
    dvalue_ddir.z += SH_C4[4] * sh[20] * (140 * z3 - 60 * z);
    dvalue_ddir.z += SH_C4[3] * sh[21] * (21 * x * z2 - 3 * x);
    dvalue_ddir.z += SH_C4[2] * sh[22] * (14 * x2 * z - 14 * y2 * z);
    dvalue_ddir.z += SH_C4[1] * sh[23] * (x3 - 3 * x * y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH5Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;

    dL_dsh[25] =
        dL_dval * SH_C5[0] * (10 * x3 * xy - 20 * x2 * y3 + 2 * y3 * y2);
    dL_dsh[26] = dL_dval * SH_C5[1] * (8 * x3 * yz - 8 * xy * y2 * z);
    dL_dsh[27] = dL_dval * SH_C5[2] *
                 (54 * x2 * y * z2 - 6 * x2 * y - 18 * y3 * z2 + 2 * y3);
    dL_dsh[28] = dL_dval * SH_C5[3] * (12 * xy * z3 - 4 * x * yz);
    dL_dsh[29] = dL_dval * SH_C5[4] * (42 * yz * z3 - 28 * y * z2 + 2 * y);
    dL_dsh[30] = dL_dval * SH_C5[5] * (63 * z2 * z3 - 70 * z3 + 15 * z);
    dL_dsh[31] = dL_dval * SH_C5[4] * (42 * x * z2 * z2 - 28 * x * z2 + 2 * x);
    dL_dsh[32] = dL_dval * SH_C5[3] *
                 (6 * x2 * z3 - 2 * x2 * z - 6 * y2 * z3 + 2 * y2 * z);
    dL_dsh[33] = dL_dval * SH_C5[2] *
                 (18 * x3 * z2 - 2 * x3 - 54 * x * y2 * z2 + 6 * x * y2);
    dL_dsh[34] = dL_dval * SH_C5[1] *
                 (2 * x2 * x2 * z - 12 * x2 * y2 * z + 2 * y2 * y2 * z);
    dL_dsh[35] =
        dL_dval * SH_C5[0] * (2 * x3 * x2 - 20 * x3 * y2 + 10 * xy * y3);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C5[0] * sh[25] * 40 * xy * (x2 - y2);
    dvalue_ddir.x += SH_C5[1] * sh[26] * 8 * yz * (3 * x2 - y2);
    dvalue_ddir.x += SH_C5[2] * sh[27] * 12 * xy * (9 * z2 - 1);
    dvalue_ddir.x += SH_C5[3] * sh[28] * 4 * yz * (3 * z2 - 1);
    dvalue_ddir.x += SH_C5[4] * sh[31] * (42 * z2 * z2 - 28 * z2 + 2);
    dvalue_ddir.x += SH_C5[3] * sh[32] * 4 * xz * (3 * z2 - 1);
    dvalue_ddir.x +=
        SH_C5[2] * sh[33] * (54 * x2 * z2 - 6 * x2 - 54 * y2 * z2 + 6 * y2);
    dvalue_ddir.x += SH_C5[1] * sh[34] * 8 * xz * (x2 - 3 * y2);
    dvalue_ddir.x +=
        SH_C5[0] * sh[35] * (10 * x2 * x2 - 60 * x2 * y2 + 10 * y2 * y2);

    dvalue_ddir.y +=
        SH_C5[0] * sh[25] * (10 * x2 * x2 - 60 * x2 * y2 + 10 * y2 * y2);
    dvalue_ddir.y += SH_C5[1] * sh[26] * 8 * xz * (x2 - 3 * y2);
    dvalue_ddir.y +=
        SH_C5[2] * sh[27] * (54 * x2 * z2 - 6 * x2 - 54 * y2 * z2 + 6 * y2);
    dvalue_ddir.y += SH_C5[3] * sh[28] * 4 * xz * (3 * z2 - 1);
    dvalue_ddir.y += SH_C5[4] * sh[29] * (42 * z2 * z2 - 28 * z2 + 2);
    dvalue_ddir.y += SH_C5[3] * sh[32] * 4 * yz * (1 - 3 * z2);
    dvalue_ddir.y += SH_C5[2] * sh[33] * 12 * xy * (1 - 9 * z2);
    dvalue_ddir.y += SH_C5[1] * sh[34] * 8 * yz * (-3 * x2 + y2);
    dvalue_ddir.y += SH_C5[0] * sh[35] * 40 * xy * (-x2 + y2);

    dvalue_ddir.z += SH_C5[1] * sh[26] * 8 * xy * (x2 - y2);
    dvalue_ddir.z += SH_C5[2] * sh[27] * 36 * yz * (3 * x2 - y2);
    dvalue_ddir.z += SH_C5[3] * sh[28] * 4 * xy * (9 * z2 - 1);
    dvalue_ddir.z += SH_C5[4] * sh[29] * 56 * yz * (3 * z2 - 1);
    dvalue_ddir.z += SH_C5[5] * sh[30] * (315 * z2 * z2 - 210 * z2 + 15);
    dvalue_ddir.z += SH_C5[4] * sh[31] * (56 * xz * (3 * z2 - 1));
    dvalue_ddir.z +=
        SH_C5[3] * sh[32] * (18 * x2 * z2 - 2 * x2 - 18 * y2 * z2 + 2 * y2);
    dvalue_ddir.z += SH_C5[2] * sh[33] * 36 * xz * (x2 - 3 * y2);
    dvalue_ddir.z +=
        SH_C5[1] * sh[34] * (2 * x2 * x2 - 12 * x2 * y2 + 2 * y2 * y2);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH6Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    dL_dsh[36] =
        dL_dval * SH_C6[0] * (12 * x3 * x2 * y - 40 * x3 * y3 + 12 * xy * y4);
    dL_dsh[37] =
        dL_dval * SH_C6[1] * (10 * x4 * yz - 20 * x2 * y2 * yz + 2 * y4 * yz);
    dL_dsh[38] =
        dL_dval * SH_C6[2] *
        (88 * x2 * xy * z2 - 8 * x2 * xy - 88 * xy * y2 * z2 + 8 * xy * y2);
    dL_dsh[39] = dL_dval * SH_C6[3] * 2 * yz *
                 (33 * x2 * z2 - 9 * x2 - 11 * y2 * z2 + 3 * y2);
    dL_dsh[40] = dL_dval * SH_C6[4] * 4 * xy * (33 * z4 - 18 * z2 + 1);
    dL_dsh[41] = dL_dval * SH_C6[5] * 2 * yz * (33 * z4 - 30 * z2 + 5);
    dL_dsh[42] = dL_dval * SH_C6[6] * (231 * z3 * z3 - 315 * z4 + 105 * z2 - 5);
    dL_dsh[43] = dL_dval * SH_C6[5] * (66 * xz * z4 - 60 * xz * z2 + 10 * xz);
    dL_dsh[44] = dL_dval * SH_C6[4] *
                 (66 * x2 * z4 - 36 * x2 * z2 + 2 * x2 - 66 * y2 * z4 +
                  36 * y2 * z2 - 2 * y2);
    dL_dsh[45] =
        dL_dval * SH_C6[3] *
        (22 * x3 * z3 - 6 * x2 * xz - 66 * xy * yz * z2 + 18 * xy * yz);
    dL_dsh[46] = dL_dval * SH_C6[2] *
                 (22 * x4 * z2 - 2 * x4 - 132 * x2 * y2 * z2 + 12 * x2 * y2 +
                  22 * y4 * z2 - 2 * y4);
    dL_dsh[47] = dL_dval * SH_C6[1] *
                 (2 * x4 * xz - 20 * x2 * xy * yz + 10 * xy * y2 * yz);
    dL_dsh[48] = dL_dval * SH_C6[0] *
                 (2 * x4 * x2 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y3 * y3);

    float3 dvalue_ddir = {0, 0, 0};
    dvalue_ddir.x += SH_C6[0] * sh[36] * 12 * y * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.x += SH_C6[1] * sh[37] * 40 * xyz * (x2 - y2);
    dvalue_ddir.x +=
        SH_C6[2] * sh[38] * 8 * y * (33 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + y2);
    dvalue_ddir.x += SH_C6[3] * sh[39] * 12 * xyz * (11 * z2 - 3);
    dvalue_ddir.x += SH_C6[4] * sh[40] * 4 * y * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.x += SH_C6[5] * sh[43] * (66 * z3 * z2 - 60 * z3 + 10 * z);
    dvalue_ddir.x += SH_C6[4] * sh[44] * 4 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.x += SH_C6[3] * sh[45] * 6 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.x +=
        SH_C6[2] * sh[46] * 8 * x * (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.x += SH_C6[1] * sh[47] * 10 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.x += SH_C6[0] * sh[48] * 12 * x * (x4 - 10 * x2 * y2 + 5 * y4);

    dvalue_ddir.y += SH_C6[0] * sh[36] * 12 * x * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.y += SH_C6[1] * sh[37] * 10 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.y +=
        SH_C6[2] * sh[38] * 8 * x * (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.y += SH_C6[3] * sh[39] * 6 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.y += SH_C6[4] * sh[40] * 4 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.y += SH_C6[5] * sh[41] * (66 * z3 * z2 - 60 * z3 + 10 * z);
    dvalue_ddir.y += SH_C6[4] * sh[44] * 4 * y * (-33 * z4 + 18 * z2 - 1);
    dvalue_ddir.y += SH_C6[3] * sh[45] * 12 * xyz * (3 - 11 * z2);
    dvalue_ddir.y += SH_C6[2] * sh[46] * 8 * y *
                     (-33 * x2 * z2 + 3 * x2 + 11 * y2 * z2 - y2);
    dvalue_ddir.y += SH_C6[1] * sh[47] * 40 * xyz * (-x2 + y2);
    dvalue_ddir.y += SH_C6[0] * sh[48] * 12 * y * (-5 * x4 + 10 * x2 * y2 - y4);

    dvalue_ddir.z += SH_C6[1] * sh[37] * 2 * y * (5 * x4 - 10 * x2 * y2 + y4);
    dvalue_ddir.z += SH_C6[2] * sh[38] * 176 * xyz * (x2 - y2);
    dvalue_ddir.z +=
        SH_C6[3] * sh[39] * 6 * y * (33 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + y2);
    dvalue_ddir.z += SH_C6[4] * sh[40] * 48 * xyz * (11 * z2 - 3);
    dvalue_ddir.z += SH_C6[5] * sh[41] * 10 * y * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.z += SH_C6[6] * sh[42] * (1386 * z3 * z2 - 1260 * z3 + 210 * z);
    dvalue_ddir.z += SH_C6[5] * sh[43] * 10 * x * (33 * z4 - 18 * z2 + 1);
    dvalue_ddir.z += SH_C6[4] * sh[44] * 24 * z *
                     (11 * x2 * z2 - 3 * x2 - 11 * y2 * z2 + 3 * y2);
    dvalue_ddir.z +=
        SH_C6[3] * sh[45] * 6 * x * (11 * x2 * z2 - x2 - 33 * y2 * z2 + 3 * y2);
    dvalue_ddir.z += SH_C6[2] * sh[46] * 44 * z * (x4 - 6 * x2 * y2 + y4);
    dvalue_ddir.z += SH_C6[1] * sh[47] * 2 * x * (x4 - 10 * x2 * y2 + 5 * y4);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH7Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x5 = x3 * x2, y5 = y3 * y2, z5 = z3 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;

    dL_dsh[49] = dL_dval * SH_C7[0] * 2 * y *
                 (7 * x3 * x3 - 35 * x4 * y2 + 21 * x2 * y4 - y3 * y3);
    dL_dsh[50] =
        dL_dval * SH_C7[1] * 4 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4);
    dL_dsh[51] = dL_dval * SH_C7[2] * 2 * y *
                 (65 * x4 * z2 - 5 * x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
                  13 * y4 * z2 - y4);
    dL_dsh[52] = dL_dval * SH_C7[3] * 8 * xyz *
                 (13 * x2 * z2 - 3 * x2 - 13 * y2 * z2 + 3 * y2);
    dL_dsh[53] = dL_dval * SH_C7[4] * 2 * y *
                 (429 * x2 * z4 - 198 * x2 * z2 + 9 * x2 - 143 * y2 * z4 +
                  66 * y2 * z2 - 3 * y2);
    dL_dsh[54] = dL_dval * SH_C7[5] * 4 * xyz * (143 * z4 - 110 * z2 + 15);
    dL_dsh[55] =
        dL_dval * SH_C7[6] * 2 * y * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    dL_dsh[56] =
        dL_dval * SH_C7[7] * z * (429 * z3 * z3 - 693 * z4 + 315 * z2 - 35);
    dL_dsh[57] =
        dL_dval * SH_C7[6] * 2 * x * (429 * z3 * z3 - 495 * z4 + 135 * z2 - 5);
    dL_dsh[58] = dL_dval * SH_C7[5] * 2 * z *
                 (143 * x2 * z4 - 110 * x2 * z2 + 15 * x2 - 143 * y2 * z4 +
                  110 * y2 * z2 - 15 * y2);
    dL_dsh[59] = dL_dval * SH_C7[4] * 2 * x *
                 (143 * x2 * z4 - 66 * x2 * z2 + 3 * x2 - 429 * y2 * z4 +
                  198 * y2 * z2 - 9 * y2);
    dL_dsh[60] = dL_dval * SH_C7[3] * 2 * z *
                 (13 * x4 * z2 - 3 * x4 - 78 * x2 * y2 * z2 + 18 * x2 * y2 +
                  13 * y4 * z2 - 3 * y4);
    dL_dsh[61] = dL_dval * SH_C7[2] * 2 * x *
                 (13 * x4 * z2 - x4 - 130 * x2 * y2 * z2 + 10 * x2 * y2 +
                  65 * y4 * z2 - 5 * y4);
    dL_dsh[62] = dL_dval * SH_C7[1] * 2 * z *
                 (x3 * x3 - 15 * x4 * y2 + 15 * x2 * y4 - y3 * y3);
    dL_dsh[63] = dL_dval * SH_C7[0] * 2 * x *
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
    dvalue_ddir.x += SH_C7[6] * sh[57] * (858 * z6 - 990 * z4 + 270 * z2 - 10);
    dvalue_ddir.x += SH_C7[5] * sh[58] * 4 * xz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.x += SH_C7[4] * sh[59] *
                     (858 * x2 * z4 - 396 * x2 * z2 + 18 * x2 - 858 * y2 * z4 +
                      396 * y2 * z2 - 18 * y2);
    dvalue_ddir.x += SH_C7[3] * sh[60] * 8 * xz *
                     (13 * x2 * z2 - 3 * x2 - 39 * y2 * z2 + 9 * y2);
    dvalue_ddir.x += SH_C7[2] * sh[61] *
                     (130 * x4 * z2 - 10 * x4 - 780 * x2 * y2 * z2 +
                      60 * x2 * y2 + 130 * y4 * z2 - 10 * y4);
    dvalue_ddir.x += SH_C7[1] * sh[62] * 12 * xz * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.x +=
        SH_C7[0] * sh[63] * (14 * x6 - 210 * x4 * y2 + 210 * x2 * y4 - 14 * y6);

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
    dvalue_ddir.y += SH_C7[5] * sh[58] * 4 * yz * (-143 * z4 + 110 * z2 - 15);
    dvalue_ddir.y += SH_C7[4] * sh[59] * 12 * xy * (-143 * z4 + 66 * z2 - 3);
    dvalue_ddir.y += SH_C7[3] * sh[60] * 8 * yz *
                     (-39 * x2 * z2 + 9 * x2 + 13 * y2 * z2 - 3 * y2);
    dvalue_ddir.y +=
        SH_C7[2] * sh[61] * 40 * xy * (-13 * x2 * z2 + x2 + 13 * y2 * z2 - y2);
    dvalue_ddir.y +=
        SH_C7[1] * sh[62] * 12 * yz * (-5 * x4 + 10 * x2 * y2 - y4);
    dvalue_ddir.y +=
        SH_C7[0] * sh[63] * (-84 * x5 * y + 280 * x3 * y3 - 84 * x * y5);

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
    dvalue_ddir.z += SH_C7[6] * sh[57] * 36 * xz * (143 * z4 - 110 * z2 + 15);
    dvalue_ddir.z += SH_C7[5] * sh[58] *
                     (1430 * x2 * z4 - 660 * x2 * z2 + 30 * x2 -
                      1430 * y2 * z4 + 660 * y2 * z2 - 30 * y2);
    dvalue_ddir.z += SH_C7[4] * sh[59] * 88 * xz *
                     (13 * x2 * z2 - 3 * x2 - 39 * y2 * z2 + 9 * y2);
    dvalue_ddir.z += SH_C7[3] * sh[60] *
                     (78 * x4 * z2 - 6 * x4 - 468 * x2 * y2 * z2 +
                      36 * x2 * y2 + 78 * y4 * z2 - 6 * y4);
    dvalue_ddir.z += SH_C7[2] * sh[61] * 52 * xz * (x4 - 10 * x2 * y2 + 5 * y4);
    dvalue_ddir.z +=
        SH_C7[1] * sh[62] * (2 * x6 - 30 * x4 * y2 + 30 * x2 * y4 - 2 * y6);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH8Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;

    dL_dsh[64] =
        dL_dval * SH_C8[0] * 16 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dL_dsh[65] = dL_dval * SH_C8[1] * 2 * yz *
                 (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dL_dsh[66] = dL_dval * SH_C8[2] * 4 * xy *
                 (45 * x4 * z2 - 3 * x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 +
                  45 * y4 * z2 - 3 * y4);
    dL_dsh[67] = dL_dval * SH_C8[3] * 2 * yz *
                 (25 * x4 * z2 - 5 * x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 +
                  5 * y4 * z2 - y4);
    dL_dsh[68] =
        dL_dval * SH_C8[4] * 8 * xy *
        (65 * x2 * z4 - 26 * x2 * z2 + x2 - 65 * y2 * z4 + 26 * y2 * z2 - y2);
    dL_dsh[69] = dL_dval * SH_C8[5] * 2 * yz *
                 (117 * x2 * z4 - 78 * x2 * z2 + 9 * x2 - 39 * y2 * z4 +
                  26 * y2 * z2 - 3 * y2);
    dL_dsh[70] =
        dL_dval * SH_C8[6] * 4 * xy * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dL_dsh[71] =
        dL_dval * SH_C8[7] * 2 * yz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    dL_dsh[72] = dL_dval * SH_C8[8] *
                 (6435 * z4 * z4 - 12012 * z6 + 6930 * z4 - 1260 * z2 + 35);
    dL_dsh[73] =
        dL_dval * SH_C8[7] * 2 * xz * (715 * z6 - 1001 * z4 + 385 * z2 - 35);
    dL_dsh[74] = dL_dval * SH_C8[6] *
                 (286 * x2 * z6 - 286 * x2 * z4 + 66 * x2 * z2 - 2 * x2 -
                  286 * y2 * z6 + 286 * y2 * z4 - 66 * y2 * z2 + 2 * y2);
    dL_dsh[75] = dL_dval * SH_C8[5] * 2 * xz *
                 (39 * x2 * z4 - 26 * x2 * z2 + 3 * x2 - 117 * y2 * z4 +
                  78 * y2 * z2 - 9 * y2);
    dL_dsh[76] = dL_dval * SH_C8[4] *
                 (130 * x4 * z4 - 52 * x4 * z2 + 2 * x4 - 780 * x2 * y2 * z4 +
                  312 * x2 * y2 * z2 - 12 * x2 * y2 + 130 * y4 * z4 -
                  52 * y4 * z2 + 2 * y4);
    dL_dsh[77] = dL_dval * SH_C8[3] * 2 * xz *
                 (5 * x4 * z2 - x4 - 50 * x2 * y2 * z2 + 10 * x2 * y2 +
                  25 * y4 * z2 - 5 * y4);
    dL_dsh[78] = dL_dval * SH_C8[2] *
                 (30 * x6 * z2 - 2 * x6 - 450 * x4 * y2 * z2 + 30 * x4 * y2 +
                  450 * x2 * y4 * z2 - 30 * x2 * y4 - 30 * y6 * z2 + 2 * y6);
    dL_dsh[79] = dL_dval * SH_C8[1] * 2 * xz *
                 (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    dL_dsh[80] = dL_dval * SH_C8[0] *
                 (2 * x4 * x4 - 56 * x6 * y2 + 140 * x4 * y4 - 56 * x2 * y6 +
                  2 * y4 * y4);

    float3 dvalue_ddir = {0, 0, 0};

    dvalue_ddir.x += SH_C8[0] * sh[64] * 16 * y *
                     (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dvalue_ddir.x +=
        SH_C8[1] * sh[65] * 28 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4);
    dvalue_ddir.x += SH_C8[2] * sh[66] * 12 * y *
                     (75 * x4 * z2 - 5 * x4 - 150 * x2 * y2 * z2 +
                      10 * x2 * y2 + 15 * y4 * z2 - y4);
    dvalue_ddir.x +=
        SH_C8[3] * sh[67] * 40 * xyz * (5 * x2 * z2 - x2 - 5 * y2 * z2 + y2);
    dvalue_ddir.x += SH_C8[4] * sh[68] * 8 * y *
                     (195 * x2 * z4 - 78 * x2 * z2 + 3 * x2 - 65 * y2 * z4 +
                      26 * y2 * z2 - y2);
    dvalue_ddir.x += SH_C8[5] * sh[69] * 12 * xyz * (39 * z4 - 26 * z2 + 3);
    dvalue_ddir.x +=
        SH_C8[6] * sh[70] * 4 * y * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dvalue_ddir.x += SH_C8[7] * sh[73] *
                     (1430 * z4 * z3 - 2002 * z3 * z2 + 770 * z3 - 70 * z);
    dvalue_ddir.x +=
        SH_C8[6] * sh[74] * 4 * x * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dvalue_ddir.x += SH_C8[5] * sh[75] * 6 * z *
                     (39 * x2 * z4 - 26 * x2 * z2 + 3 * x2 - 39 * y2 * z4 +
                      26 * y2 * z2 - 3 * y2);
    dvalue_ddir.x += SH_C8[4] * sh[76] * 8 * x *
                     (65 * x2 * z4 - 26 * x2 * z2 + x2 - 195 * y2 * z4 +
                      78 * y2 * z2 - 3 * y2);
    dvalue_ddir.x +=
        SH_C8[3] * sh[77] * 10 * z *
        (5 * x4 * z2 - x4 - 30 * x2 * y2 * z2 + 6 * x2 * y2 + 5 * y4 * z2 - y4);
    dvalue_ddir.x += SH_C8[2] * sh[78] * 12 * x *
                     (15 * x4 * z2 - x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 +
                      75 * y4 * z2 - 5 * y4);
    dvalue_ddir.x +=
        SH_C8[1] * sh[79] * 14 * z * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6);
    dvalue_ddir.x += SH_C8[0] * sh[80] * 16 * x *
                     (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);

    dvalue_ddir.y += SH_C8[0] * sh[64] * 16 * x *
                     (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    dvalue_ddir.y +=
        SH_C8[1] * sh[65] * 14 * z * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6);
    dvalue_ddir.y += SH_C8[2] * sh[66] * 12 * x *
                     (15 * x4 * z2 - x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 +
                      75 * y4 * z2 - 5 * y4);
    dvalue_ddir.y +=
        SH_C8[3] * sh[67] * 10 * z *
        (5 * x4 * z2 - x4 - 30 * x2 * y2 * z2 + 6 * x2 * y2 + 5 * y4 * z2 - y4);
    dvalue_ddir.y += SH_C8[4] * sh[68] * 8 * x *
                     (65 * x2 * z4 - 26 * x2 * z2 + x2 - 195 * y2 * z4 +
                      78 * y2 * z2 - 3 * y2);
    dvalue_ddir.y += SH_C8[5] * sh[69] * 6 * z *
                     (39 * x2 * z4 - 26 * x2 * z2 + 3 * x2 - 39 * y2 * z4 +
                      26 * y2 * z2 - 3 * y2);
    dvalue_ddir.y +=
        SH_C8[6] * sh[70] * 4 * x * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dvalue_ddir.y += SH_C8[7] * sh[71] *
                     (1430 * z4 * z3 - 2002 * z3 * z2 + 770 * z3 - 70 * z);
    dvalue_ddir.y +=
        SH_C8[6] * sh[74] * 4 * y * (-143 * z6 + 143 * z4 - 33 * z2 + 1);
    dvalue_ddir.y += SH_C8[5] * sh[75] * 12 * xyz * (-39 * z4 + 26 * z2 - 3);
    dvalue_ddir.y += SH_C8[4] * sh[76] * 8 * y *
                     (-195 * x2 * z4 + 78 * x2 * z2 - 3 * x2 + 65 * y2 * z4 -
                      26 * y2 * z2 + y2);
    dvalue_ddir.y +=
        SH_C8[3] * sh[77] * 40 * xyz * (-5 * x2 * z2 + x2 + 5 * y2 * z2 - y2);
    dvalue_ddir.y += SH_C8[2] * sh[78] * 12 * y *
                     (-75 * x4 * z2 + 5 * x4 + 150 * x2 * y2 * z2 -
                      10 * x2 * y2 - 15 * y4 * z2 + y4);
    dvalue_ddir.y +=
        SH_C8[1] * sh[79] * 28 * xyz * (-3 * x4 + 10 * x2 * y2 - 3 * y4);
    dvalue_ddir.y += SH_C8[0] * sh[80] * 16 * y *
                     (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6);

    dvalue_ddir.z +=
        SH_C8[1] * sh[65] * 2 * y * (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dvalue_ddir.z +=
        SH_C8[2] * sh[66] * 120 * xyz * (3 * x4 - 10 * x2 * y2 + 3 * y4);
    dvalue_ddir.z += SH_C8[3] * sh[67] * 2 * y *
                     (75 * x4 * z2 - 5 * x4 - 150 * x2 * y2 * z2 +
                      10 * x2 * y2 + 15 * y4 * z2 - y4);
    dvalue_ddir.z +=
        SH_C8[4] * sh[68] * 416 * xyz * (5 * x2 * z2 - x2 - 5 * y2 * z2 + y2);
    dvalue_ddir.z += SH_C8[5] * sh[69] * 6 * y *
                     (195 * x2 * z4 - 78 * x2 * z2 + 3 * x2 - 65 * y2 * z4 +
                      26 * y2 * z2 - y2);
    dvalue_ddir.z += SH_C8[6] * sh[70] * 88 * xyz * (39 * z4 - 26 * z2 + 3);
    dvalue_ddir.z +=
        SH_C8[7] * sh[71] * 70 * y * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dvalue_ddir.z +=
        SH_C8[8] * sh[72] *
        (51480 * z4 * z3 - 72072 * z3 * z2 + 27720 * z3 - 2520 * z);
    dvalue_ddir.z +=
        SH_C8[7] * sh[73] * 70 * x * (143 * z6 - 143 * z4 + 33 * z2 - 1);
    dvalue_ddir.z += SH_C8[6] * sh[74] * 44 * z *
                     (39 * x2 * z4 - 26 * x2 * z2 + 3 * x2 - 39 * y2 * z4 +
                      26 * y2 * z2 - 3 * y2);
    dvalue_ddir.z += SH_C8[5] * sh[75] * 6 * x *
                     (65 * x2 * z4 - 26 * x2 * z2 + x2 - 195 * y2 * z4 +
                      78 * y2 * z2 - 3 * y2);
    dvalue_ddir.z +=
        SH_C8[4] * sh[76] * 104 * z *
        (5 * x4 * z2 - x4 - 30 * x2 * y2 * z2 + 6 * x2 * y2 + 5 * y4 * z2 - y4);
    dvalue_ddir.z += SH_C8[3] * sh[77] * 2 * x *
                     (15 * x4 * z2 - x4 - 150 * x2 * y2 * z2 + 10 * x2 * y2 +
                      75 * y4 * z2 - 5 * y4);
    dvalue_ddir.z +=
        SH_C8[2] * sh[78] * 60 * z * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6);
    dvalue_ddir.z +=
        SH_C8[1] * sh[79] * 2 * x * (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH9Backward(const float *sh,
                                                    const float3 dir,
                                                    const float dL_dval,
                                                    float *dL_dsh,
                                                    float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;
    float x8 = x4 * x4, y8 = y4 * y4, z8 = z4 * z4;

    dL_dsh[81] = dL_dval * SH_C9[0] * 2 * y *
                 (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    dL_dsh[82] =
        dL_dval * SH_C9[1] * 16 * xyz * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dL_dsh[83] = dL_dval * SH_C9[2] * 2 * y *
                 (119 * x6 * z2 - 7 * x6 - 595 * x4 * y2 * z2 + 35 * x4 * y2 +
                  357 * x2 * y4 * z2 - 21 * x2 * y4 - 17 * y6 * z2 + y6);
    dL_dsh[84] = dL_dval * SH_C9[3] * 4 * xyz *
                 (51 * x4 * z2 - 9 * x4 - 170 * x2 * y2 * z2 + 30 * x2 * y2 +
                  51 * y4 * z2 - 9 * y4);
    dL_dsh[85] =
        dL_dval * SH_C9[4] * 2 * y *
        (425 * x4 * z4 - 150 * x4 * z2 + 5 * x4 - 850 * x2 * y2 * z4 +
         300 * x2 * y2 * z2 - 10 * x2 * y2 + 85 * y4 * z4 - 30 * y4 * z2 + y4);
    dL_dsh[86] =
        dL_dval * SH_C9[5] * 8 * xyz *
        (17 * x2 * z4 - 10 * x2 * z2 + x2 - 17 * y2 * z4 + 10 * y2 * z2 - y2);
    dL_dsh[87] = dL_dval * SH_C9[6] * 2 * y *
                 (663 * x2 * z6 - 585 * x2 * z4 + 117 * x2 * z2 - 3 * x2 -
                  221 * y2 * z6 + 195 * y2 * z4 - 39 * y2 * z2 + y2);
    dL_dsh[88] =
        dL_dval * SH_C9[7] * 4 * xyz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dL_dsh[89] = dL_dval * SH_C9[8] * 2 * y *
                 (2431 * z8 - 4004 * z6 + 2002 * z4 - 308 * z2 + 7);
    dL_dsh[90] = dL_dval * SH_C9[9] * z *
                 (12155 * z8 - 25740 * z6 + 18018 * z4 - 4620 * z2 + 315);
    dL_dsh[91] = dL_dval * SH_C9[8] * 2 * x *
                 (2431 * z8 - 4004 * z6 + 2002 * z4 - 308 * z2 + 7);
    dL_dsh[92] = dL_dval * SH_C9[7] * 2 * z *
                 (221 * x2 * z6 - 273 * x2 * z4 + 91 * x2 * z2 - 7 * x2 -
                  221 * y2 * z6 + 273 * y2 * z4 - 91 * y2 * z2 + 7 * y2);
    dL_dsh[93] = dL_dval * SH_C9[6] * 2 * x *
                 (221 * x2 * z6 - 195 * x2 * z4 + 39 * x2 * z2 - x2 -
                  663 * y2 * z6 + 585 * y2 * z4 - 117 * y2 * z2 + 3 * y2);
    dL_dsh[94] =
        dL_dval * SH_C9[5] * 2 * z *
        (17 * x4 * z4 - 10 * x4 * z2 + x4 - 102 * x2 * y2 * z4 +
         60 * x2 * y2 * z2 - 6 * x2 * y2 + 17 * y4 * z4 - 10 * y4 * z2 + y4);
    dL_dsh[95] = dL_dval * SH_C9[4] * 2 * x *
                 (85 * x4 * z4 - 30 * x4 * z2 + x4 - 850 * x2 * y2 * z4 +
                  300 * x2 * y2 * z2 - 10 * x2 * y2 + 425 * y4 * z4 -
                  150 * y4 * z2 + 5 * y4);
    dL_dsh[96] = dL_dval * SH_C9[3] * 2 * z *
                 (17 * x6 * z2 - 3 * x6 - 255 * x4 * y2 * z2 + 45 * x4 * y2 +
                  255 * x2 * y4 * z2 - 45 * x2 * y4 - 17 * y6 * z2 + 3 * y6);
    dL_dsh[97] = dL_dval * SH_C9[2] * 2 * x *
                 (17 * x6 * z2 - x6 - 357 * x4 * y2 * z2 + 21 * x4 * y2 +
                  595 * x2 * y4 * z2 - 35 * x2 * y4 - 119 * y6 * z2 + 7 * y6);
    dL_dsh[98] = dL_dval * SH_C9[1] * 2 * z *
                 (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8);
    dL_dsh[99] = dL_dval * SH_C9[0] * 2 * x *
                 (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);

    float3 dvalue_ddir = {0, 0, 0};

    dvalue_ddir.x +=
        SH_C9[0] * sh[81] * 144 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dvalue_ddir.x += SH_C9[1] * sh[82] * 16 * yz *
                     (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dvalue_ddir.x += SH_C9[2] * sh[83] * 28 * xy *
                     (51 * x4 * z2 - 3 * x4 - 170 * x2 * y2 * z2 +
                      10 * x2 * y2 + 51 * y4 * z2 - 3 * y4);
    dvalue_ddir.x += SH_C9[3] * sh[84] * 12 * yz *
                     (85 * x4 * z2 - 15 * x4 - 170 * x2 * y2 * z2 +
                      30 * x2 * y2 + 17 * y4 * z2 - 3 * y4);
    dvalue_ddir.x +=
        SH_C9[4] * sh[85] * 40 * xy *
        (85 * x2 * z4 - 30 * x2 * z2 + x2 - 85 * y2 * z4 + 30 * y2 * z2 - y2);
    dvalue_ddir.x += SH_C9[5] * sh[86] * 8 * yz *
                     (51 * x2 * z4 - 30 * x2 * z2 + 3 * x2 - 17 * y2 * z4 +
                      10 * y2 * z2 - y2);
    dvalue_ddir.x +=
        SH_C9[6] * sh[87] * 12 * xy * (221 * z6 - 195 * z4 + 39 * z2 - 1);
    dvalue_ddir.x +=
        SH_C9[7] * sh[88] * 4 * yz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dvalue_ddir.x +=
        SH_C9[8] * sh[91] * (4862 * z8 - 8008 * z6 + 4004 * z4 - 616 * z2 + 14);
    dvalue_ddir.x +=
        SH_C9[7] * sh[92] * 4 * xz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dvalue_ddir.x += SH_C9[6] * sh[93] *
                     (1326 * x2 * z6 - 1170 * x2 * z4 + 234 * x2 * z2 - 6 * x2 -
                      1326 * y2 * z6 + 1170 * y2 * z4 - 234 * y2 * z2 + 6 * y2);
    dvalue_ddir.x += SH_C9[5] * sh[94] * 8 * xz *
                     (17 * x2 * z4 - 10 * x2 * z2 + x2 - 51 * y2 * z4 +
                      30 * y2 * z2 - 3 * y2);
    dvalue_ddir.x += SH_C9[4] * sh[95] *
                     (850 * x4 * z4 - 300 * x4 * z2 + 10 * x4 -
                      5100 * x2 * y2 * z4 + 1800 * x2 * y2 * z2 - 60 * x2 * y2 +
                      850 * y4 * z4 - 300 * y4 * z2 + 10 * y4);
    dvalue_ddir.x += SH_C9[3] * sh[96] * 12 * xz *
                     (17 * x4 * z2 - 3 * x4 - 170 * x2 * y2 * z2 +
                      30 * x2 * y2 + 85 * y4 * z2 - 15 * y4);
    dvalue_ddir.x +=
        SH_C9[2] * sh[97] *
        (238 * x6 * z2 - 14 * x6 - 3570 * x4 * y2 * z2 + 210 * x4 * y2 +
         3570 * x2 * y4 * z2 - 210 * x2 * y4 - 238 * y6 * z2 + 14 * y6);
    dvalue_ddir.x += SH_C9[1] * sh[98] * 16 * xz *
                     (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    dvalue_ddir.x +=
        SH_C9[0] * sh[99] *
        (18 * x8 - 504 * x6 * y2 + 1260 * x4 * y4 - 504 * x2 * y6 + 18 * y8);

    dvalue_ddir.y +=
        SH_C9[0] * sh[81] *
        (18 * x8 - 504 * x6 * y2 + 1260 * x4 * y4 - 504 * x2 * y6 + 18 * y8);
    dvalue_ddir.y += SH_C9[1] * sh[82] * 16 * xz *
                     (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    dvalue_ddir.y +=
        SH_C9[2] * sh[83] *
        (238 * x6 * z2 - 14 * x6 - 3570 * x4 * y2 * z2 + 210 * x4 * y2 +
         3570 * x2 * y4 * z2 - 210 * x2 * y4 - 238 * y6 * z2 + 14 * y6);
    dvalue_ddir.y += SH_C9[3] * sh[84] * 12 * xz *
                     (17 * x4 * z2 - 3 * x4 - 170 * x2 * y2 * z2 +
                      30 * x2 * y2 + 85 * y4 * z2 - 15 * y4);
    dvalue_ddir.y += SH_C9[4] * sh[85] *
                     (850 * x4 * z4 - 300 * x4 * z2 + 10 * x4 -
                      5100 * x2 * y2 * z4 + 1800 * x2 * y2 * z2 - 60 * x2 * y2 +
                      850 * y4 * z4 - 300 * y4 * z2 + 10 * y4);
    dvalue_ddir.y += SH_C9[5] * sh[86] * 8 * xz *
                     (17 * x2 * z4 - 10 * x2 * z2 + x2 - 51 * y2 * z4 +
                      30 * y2 * z2 - 3 * y2);
    dvalue_ddir.y += SH_C9[6] * sh[87] *
                     (1326 * x2 * z6 - 1170 * x2 * z4 + 234 * x2 * z2 - 6 * x2 -
                      1326 * y2 * z6 + 1170 * y2 * z4 - 234 * y2 * z2 + 6 * y2);
    dvalue_ddir.y +=
        SH_C9[7] * sh[88] * 4 * xz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dvalue_ddir.y +=
        SH_C9[8] * sh[89] * (4862 * z8 - 8008 * z6 + 4004 * z4 - 616 * z2 + 14);
    dvalue_ddir.y +=
        SH_C9[7] * sh[92] * 4 * yz * (-221 * z6 + 273 * z4 - 91 * z2 + 7);
    dvalue_ddir.y +=
        SH_C9[6] * sh[93] * 12 * xy * (-221 * z6 + 195 * z4 - 39 * z2 + 1);
    dvalue_ddir.y += SH_C9[5] * sh[94] * 8 * yz *
                     (-51 * x2 * z4 + 30 * x2 * z2 - 3 * x2 + 17 * y2 * z4 -
                      10 * y2 * z2 + y2);
    dvalue_ddir.y +=
        SH_C9[4] * sh[95] * 40 * xy *
        (-85 * x2 * z4 + 30 * x2 * z2 - x2 + 85 * y2 * z4 - 30 * y2 * z2 + y2);
    dvalue_ddir.y += SH_C9[3] * sh[96] * 12 * yz *
                     (-85 * x4 * z2 + 15 * x4 + 170 * x2 * y2 * z2 -
                      30 * x2 * y2 - 17 * y4 * z2 + 3 * y4);
    dvalue_ddir.y += SH_C9[2] * sh[97] * 28 * xy *
                     (-51 * x4 * z2 + 3 * x4 + 170 * x2 * y2 * z2 -
                      10 * x2 * y2 - 51 * y4 * z2 + 3 * y4);
    dvalue_ddir.y += SH_C9[1] * sh[98] * 16 * yz *
                     (-7 * x6 + 35 * x4 * y2 - 21 * x2 * y4 + y6);
    dvalue_ddir.y +=
        SH_C9[0] * sh[99] * 144 * xy * (-x6 + 7 * x4 * y2 - 7 * x2 * y4 + y6);

    dvalue_ddir.z +=
        SH_C9[1] * sh[82] * 16 * xy * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dvalue_ddir.z += SH_C9[2] * sh[83] * 68 * yz *
                     (7 * x6 - 35 * x4 * y2 + 21 * x2 * y4 - y6);
    dvalue_ddir.z += SH_C9[3] * sh[84] * 12 * xy *
                     (51 * x4 * z2 - 3 * x4 - 170 * x2 * y2 * z2 +
                      10 * x2 * y2 + 51 * y4 * z2 - 3 * y4);
    dvalue_ddir.z += SH_C9[4] * sh[85] * 40 * yz *
                     (85 * x4 * z2 - 15 * x4 - 170 * x2 * y2 * z2 +
                      30 * x2 * y2 + 17 * y4 * z2 - 3 * y4);
    dvalue_ddir.z +=
        SH_C9[5] * sh[86] * 8 * xy *
        (85 * x2 * z4 - 30 * x2 * z2 + x2 - 85 * y2 * z4 + 30 * y2 * z2 - y2);
    dvalue_ddir.z += SH_C9[6] * sh[87] * 156 * yz *
                     (51 * x2 * z4 - 30 * x2 * z2 + 3 * x2 - 17 * y2 * z4 +
                      10 * y2 * z2 - y2);
    dvalue_ddir.z +=
        SH_C9[7] * sh[88] * 28 * xy * (221 * z6 - 195 * z4 + 39 * z2 - 1);
    dvalue_ddir.z +=
        SH_C9[8] * sh[89] * 176 * yz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dvalue_ddir.z +=
        SH_C9[9] * sh[90] *
        (109395 * z8 - 180180 * z6 + 90090 * z4 - 13860 * z2 + 315);
    dvalue_ddir.z +=
        SH_C9[8] * sh[91] * 176 * xz * (221 * z6 - 273 * z4 + 91 * z2 - 7);
    dvalue_ddir.z +=
        SH_C9[7] * sh[92] *
        (3094 * x2 * z6 - 2730 * x2 * z4 + 546 * x2 * z2 - 14 * x2 -
         3094 * y2 * z6 + 2730 * y2 * z4 - 546 * y2 * z2 + 14 * y2);
    dvalue_ddir.z += SH_C9[6] * sh[93] * 156 * xz *
                     (17 * x2 * z4 - 10 * x2 * z2 + x2 - 51 * y2 * z4 +
                      30 * y2 * z2 - 3 * y2);
    dvalue_ddir.z += SH_C9[5] * sh[94] *
                     (170 * x4 * z4 - 60 * x4 * z2 + 2 * x4 -
                      1020 * x2 * y2 * z4 + 360 * x2 * y2 * z2 - 12 * x2 * y2 +
                      170 * y4 * z4 - 60 * y4 * z2 + 2 * y4);
    dvalue_ddir.z += SH_C9[4] * sh[95] * 40 * xz *
                     (17 * x4 * z2 - 3 * x4 - 170 * x2 * y2 * z2 +
                      30 * x2 * y2 + 85 * y4 * z2 - 15 * y4);
    dvalue_ddir.z +=
        SH_C9[3] * sh[96] *
        (102 * x6 * z2 - 6 * x6 - 1530 * x4 * y2 * z2 + 90 * x4 * y2 +
         1530 * x2 * y4 * z2 - 90 * x2 * y4 - 102 * y6 * z2 + 6 * y6);
    dvalue_ddir.z += SH_C9[2] * sh[97] * 68 * xz *
                     (x6 - 21 * x4 * y2 + 35 * x2 * y4 - 7 * y6);
    dvalue_ddir.z +=
        SH_C9[1] * sh[98] *
        (2 * x8 - 56 * x6 * y2 + 140 * x4 * y4 - 56 * x2 * y6 + 2 * y8);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
}

__forceinline__ __device__ void evaluateSH10Backward(const float *sh,
                                                     const float3 dir,
                                                     const float dL_dval,
                                                     float *dL_dsh,
                                                     float3 *dL_ddir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    float x3 = x * x2, y3 = y * y2, z3 = z * z2, xyz = x * y * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x3 * x3, y6 = y3 * y3, z6 = z3 * z3;
    float x8 = x4 * x4, y8 = y4 * y4, z8 = z4 * z4;
    float x10 = x6 * x4, y10 = y6 * y4, z10 = z6 * z4;

    dL_dsh[100] =
        dL_dval * SH_C10[0] * xy *
        (20 * x8 - 240 * x6 * y2 + 504 * x4 * y4 - 240 * x2 * y6 + 20 * y8);
    dL_dsh[101] = dL_dval * SH_C10[1] * 2 * yz *
                  (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    dL_dsh[102] = dL_dval * SH_C10[2] * 16 * xy *
                  (19 * x6 * z2 - x6 - 133 * x4 * y2 * z2 + 7 * x4 * y2 +
                   133 * x2 * y4 * z2 - 7 * x2 * y4 - 19 * y6 * z2 + y6);
    dL_dsh[103] =
        dL_dval * SH_C10[3] * 2 * yz *
        (133 * x6 * z2 - 21 * x6 - 665 * x4 * y2 * z2 + 105 * x4 * y2 +
         399 * x2 * y4 * z2 - 63 * x2 * y4 - 19 * y6 * z2 + 3 * y6);
    dL_dsh[104] = dL_dval * SH_C10[4] * 4 * xy *
                  (969 * x4 * z4 - 306 * x4 * z2 + 9 * x4 -
                   3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                   969 * y4 * z4 - 306 * y4 * z2 + 9 * y4);
    dL_dsh[105] = dL_dval * SH_C10[5] * 2 * yz *
                  (1615 * x4 * z4 - 850 * x4 * z2 + 75 * x4 -
                   3230 * x2 * y2 * z4 + 1700 * x2 * y2 * z2 - 150 * x2 * y2 +
                   323 * y4 * z4 - 170 * y4 * z2 + 15 * y4);
    dL_dsh[106] = dL_dval * SH_C10[6] * 8 * xy *
                  (323 * x2 * z6 - 255 * x2 * z4 + 45 * x2 * z2 - x2 -
                   323 * y2 * z6 + 255 * y2 * z4 - 45 * y2 * z2 + y2);
    dL_dsh[107] = dL_dval * SH_C10[7] * 2 * yz *
                  (969 * x2 * z6 - 1071 * x2 * z4 + 315 * x2 * z2 - 21 * x2 -
                   323 * y2 * z6 + 357 * y2 * z4 - 105 * y2 * z2 + 7 * y2);
    dL_dsh[108] = dL_dval * SH_C10[8] * 4 * xy *
                  (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dL_dsh[109] = dL_dval * SH_C10[9] * 2 * yz *
                  (4199 * z8 - 7956 * z6 + 4914 * z4 - 1092 * z2 + 63);
    dL_dsh[110] =
        dL_dval * SH_C10[10] *
        (46189 * z10 - 109395 * z8 + 90090 * z6 - 30030 * z4 + 3465 * z2 - 63);
    dL_dsh[111] = dL_dval * SH_C10[9] * 2 * xz *
                  (4199 * z8 - 7956 * z6 + 4914 * z4 - 1092 * z2 + 63);
    dL_dsh[112] = dL_dval * SH_C10[8] *
                  (8398 * x2 * z8 - 12376 * x2 * z6 + 5460 * x2 * z4 -
                   728 * x2 * z2 + 14 * x2 - 8398 * y2 * z8 + 12376 * y2 * z6 -
                   5460 * y2 * z4 + 728 * y2 * z2 - 14 * y2);
    dL_dsh[113] = dL_dval * SH_C10[7] * 2 * xz *
                  (323 * x2 * z6 - 357 * x2 * z4 + 105 * x2 * z2 - 7 * x2 -
                   969 * y2 * z6 + 1071 * y2 * z4 - 315 * y2 * z2 + 21 * y2);
    dL_dsh[114] =
        dL_dval * SH_C10[6] *
        (646 * x4 * z6 - 510 * x4 * z4 + 90 * x4 * z2 - 2 * x4 -
         3876 * x2 * y2 * z6 + 3060 * x2 * y2 * z4 - 540 * x2 * y2 * z2 +
         12 * x2 * y2 + 646 * y4 * z6 - 510 * y4 * z4 + 90 * y4 * z2 - 2 * y4);
    dL_dsh[115] = dL_dval * SH_C10[5] * 2 * xz *
                  (323 * x4 * z4 - 170 * x4 * z2 + 15 * x4 -
                   3230 * x2 * y2 * z4 + 1700 * x2 * y2 * z2 - 150 * x2 * y2 +
                   1615 * y4 * z4 - 850 * y4 * z2 + 75 * y4);
    dL_dsh[116] = dL_dval * SH_C10[4] *
                  (646 * x6 * z4 - 204 * x6 * z2 + 6 * x6 -
                   9690 * x4 * y2 * z4 + 3060 * x4 * y2 * z2 - 90 * x4 * y2 +
                   9690 * x2 * y4 * z4 - 3060 * x2 * y4 * z2 + 90 * x2 * y4 -
                   646 * y6 * z4 + 204 * y6 * z2 - 6 * y6);
    dL_dsh[117] =
        dL_dval * SH_C10[3] * 2 * xz *
        (19 * x6 * z2 - 3 * x6 - 399 * x4 * y2 * z2 + 63 * x4 * y2 +
         665 * x2 * y4 * z2 - 105 * x2 * y4 - 133 * y6 * z2 + 21 * y6);
    dL_dsh[118] = dL_dval * SH_C10[2] *
                  (38 * x8 * z2 - 2 * x8 - 1064 * x6 * y2 * z2 + 56 * x6 * y2 +
                   2660 * x4 * y4 * z2 - 140 * x4 * y4 - 1064 * x2 * y6 * z2 +
                   56 * x2 * y6 + 38 * y8 * z2 - 2 * y8);
    dL_dsh[119] = dL_dval * SH_C10[1] * 2 * xz *
                  (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);
    dL_dsh[120] = dL_dval * SH_C10[0] *
                  (2 * x10 - 90 * x8 * y2 + 420 * x6 * y4 - 420 * x4 * y6 +
                   90 * x2 * y8 - 2 * y10);

    float3 dvalue_ddir = {0, 0, 0};

    dvalue_ddir.x +=
        SH_C10[0] * sh[100] * 20 * y *
        (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    dvalue_ddir.x +=
        SH_C10[1] * sh[101] * 144 * xyz * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dvalue_ddir.x +=
        SH_C10[2] * sh[102] * 16 * y *
        (133 * x6 * z2 - 7 * x6 - 665 * x4 * y2 * z2 + 35 * x4 * y2 +
         399 * x2 * y4 * z2 - 21 * x2 * y4 - 19 * y6 * z2 + y6);
    dvalue_ddir.x += SH_C10[3] * sh[103] * 28 * xyz *
                     (57 * x4 * z2 - 9 * x4 - 190 * x2 * y2 * z2 +
                      30 * x2 * y2 + 57 * y4 * z2 - 9 * y4);
    dvalue_ddir.x += SH_C10[4] * sh[104] * 12 * y *
                     (1615 * x4 * z4 - 510 * x4 * z2 + 15 * x4 -
                      3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                      323 * y4 * z4 - 102 * y4 * z2 + 3 * y4);
    dvalue_ddir.x += SH_C10[5] * sh[105] * 40 * xyz *
                     (323 * x2 * z4 - 170 * x2 * z2 + 15 * x2 - 323 * y2 * z4 +
                      170 * y2 * z2 - 15 * y2);
    dvalue_ddir.x += SH_C10[6] * sh[106] * 8 * y *
                     (969 * x2 * z6 - 765 * x2 * z4 + 135 * x2 * z2 - 3 * x2 -
                      323 * y2 * z6 + 255 * y2 * z4 - 45 * y2 * z2 + y2);
    dvalue_ddir.x +=
        SH_C10[7] * sh[107] * 12 * xyz * (323 * z6 - 357 * z4 + 105 * z2 - 7);
    dvalue_ddir.x += SH_C10[8] * sh[108] * 4 * y *
                     (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dvalue_ddir.x += SH_C10[9] * sh[111] * z *
                     (8398 * z8 - 15912 * z6 + 9828 * z4 - 2184 * z2 + 126);
    dvalue_ddir.x += SH_C10[8] * sh[112] * 4 * x *
                     (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dvalue_ddir.x += SH_C10[7] * sh[113] * 6 * z *
                     (323 * x2 * z6 - 357 * x2 * z4 + 105 * x2 * z2 - 7 * x2 -
                      323 * y2 * z6 + 357 * y2 * z4 - 105 * y2 * z2 + 7 * y2);
    dvalue_ddir.x += SH_C10[6] * sh[114] * 8 * x *
                     (323 * x2 * z6 - 255 * x2 * z4 + 45 * x2 * z2 - x2 -
                      969 * y2 * z6 + 765 * y2 * z4 - 135 * y2 * z2 + 3 * y2);
    dvalue_ddir.x += SH_C10[5] * sh[115] * 10 * z *
                     (323 * x4 * z4 - 170 * x4 * z2 + 15 * x4 -
                      1938 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 90 * x2 * y2 +
                      323 * y4 * z4 - 170 * y4 * z2 + 15 * y4);
    dvalue_ddir.x += SH_C10[4] * sh[116] * 12 * x *
                     (323 * x4 * z4 - 102 * x4 * z2 + 3 * x4 -
                      3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                      1615 * y4 * z4 - 510 * y4 * z2 + 15 * y4);
    dvalue_ddir.x +=
        SH_C10[3] * sh[117] * 14 * z *
        (19 * x6 * z2 - 3 * x6 - 285 * x4 * y2 * z2 + 45 * x4 * y2 +
         285 * x2 * y4 * z2 - 45 * x2 * y4 - 19 * y6 * z2 + 3 * y6);
    dvalue_ddir.x +=
        SH_C10[2] * sh[118] * 16 * x *
        (19 * x6 * z2 - x6 - 399 * x4 * y2 * z2 + 21 * x4 * y2 +
         665 * x2 * y4 * z2 - 35 * x2 * y4 - 133 * y6 * z2 + 7 * y6);
    dvalue_ddir.x += SH_C10[1] * sh[119] * 18 * z *
                     (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8);
    dvalue_ddir.x +=
        SH_C10[0] * sh[120] * 20 * x *
        (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);

    dvalue_ddir.y +=
        SH_C10[0] * sh[100] * 20 * x *
        (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);
    dvalue_ddir.y += SH_C10[1] * sh[101] * 18 * z *
                     (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8);
    dvalue_ddir.y +=
        SH_C10[2] * sh[102] * 16 * x *
        (19 * x6 * z2 - x6 - 399 * x4 * y2 * z2 + 21 * x4 * y2 +
         665 * x2 * y4 * z2 - 35 * x2 * y4 - 133 * y6 * z2 + 7 * y6);
    dvalue_ddir.y +=
        SH_C10[3] * sh[103] * 14 * z *
        (19 * x6 * z2 - 3 * x6 - 285 * x4 * y2 * z2 + 45 * x4 * y2 +
         285 * x2 * y4 * z2 - 45 * x2 * y4 - 19 * y6 * z2 + 3 * y6);
    dvalue_ddir.y += SH_C10[4] * sh[104] * 12 * x *
                     (323 * x4 * z4 - 102 * x4 * z2 + 3 * x4 -
                      3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                      1615 * y4 * z4 - 510 * y4 * z2 + 15 * y4);
    dvalue_ddir.y += SH_C10[5] * sh[105] * 10 * z *
                     (323 * x4 * z4 - 170 * x4 * z2 + 15 * x4 -
                      1938 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 90 * x2 * y2 +
                      323 * y4 * z4 - 170 * y4 * z2 + 15 * y4);
    dvalue_ddir.y += SH_C10[6] * sh[106] * 8 * x *
                     (323 * x2 * z6 - 255 * x2 * z4 + 45 * x2 * z2 - x2 -
                      969 * y2 * z6 + 765 * y2 * z4 - 135 * y2 * z2 + 3 * y2);
    dvalue_ddir.y += SH_C10[7] * sh[107] * 6 * z *
                     (323 * x2 * z6 - 357 * x2 * z4 + 105 * x2 * z2 - 7 * x2 -
                      323 * y2 * z6 + 357 * y2 * z4 - 105 * y2 * z2 + 7 * y2);
    dvalue_ddir.y += SH_C10[8] * sh[108] * 4 * x *
                     (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dvalue_ddir.y += SH_C10[9] * sh[109] * z *
                     (8398 * z8 - 15912 * z6 + 9828 * z4 - 2184 * z2 + 126);
    dvalue_ddir.y += SH_C10[8] * sh[112] * 4 * y *
                     (-4199 * z8 + 6188 * z6 - 2730 * z4 + 364 * z2 - 7);
    dvalue_ddir.y +=
        SH_C10[7] * sh[113] * 12 * xyz * (-323 * z6 + 357 * z4 - 105 * z2 + 7);
    dvalue_ddir.y += SH_C10[6] * sh[114] * 8 * y *
                     (-969 * x2 * z6 + 765 * x2 * z4 - 135 * x2 * z2 + 3 * x2 +
                      323 * y2 * z6 - 255 * y2 * z4 + 45 * y2 * z2 - y2);
    dvalue_ddir.y += SH_C10[5] * sh[115] * 40 * xyz *
                     (-323 * x2 * z4 + 170 * x2 * z2 - 15 * x2 + 323 * y2 * z4 -
                      170 * y2 * z2 + 15 * y2);
    dvalue_ddir.y += SH_C10[4] * sh[116] * 12 * y *
                     (-1615 * x4 * z4 + 510 * x4 * z2 - 15 * x4 +
                      3230 * x2 * y2 * z4 - 1020 * x2 * y2 * z2 + 30 * x2 * y2 -
                      323 * y4 * z4 + 102 * y4 * z2 - 3 * y4);
    dvalue_ddir.y += SH_C10[3] * sh[117] * 28 * xyz *
                     (-57 * x4 * z2 + 9 * x4 + 190 * x2 * y2 * z2 -
                      30 * x2 * y2 - 57 * y4 * z2 + 9 * y4);
    dvalue_ddir.y +=
        SH_C10[2] * sh[118] * 16 * y *
        (-133 * x6 * z2 + 7 * x6 + 665 * x4 * y2 * z2 - 35 * x4 * y2 -
         399 * x2 * y4 * z2 + 21 * x2 * y4 + 19 * y6 * z2 - y6);
    dvalue_ddir.y += SH_C10[1] * sh[119] * 144 * xyz *
                     (-x6 + 7 * x4 * y2 - 7 * x2 * y4 + y6);
    dvalue_ddir.y +=
        SH_C10[0] * sh[120] * 20 * y *
        (-9 * x8 + 84 * x6 * y2 - 126 * x4 * y4 + 36 * x2 * y6 - y8);

    dvalue_ddir.z +=
        SH_C10[1] * sh[101] * 2 * y *
        (9 * x8 - 84 * x6 * y2 + 126 * x4 * y4 - 36 * x2 * y6 + y8);
    dvalue_ddir.z +=
        SH_C10[2] * sh[102] * 608 * xyz * (x6 - 7 * x4 * y2 + 7 * x2 * y4 - y6);
    dvalue_ddir.z +=
        SH_C10[3] * sh[103] * 6 * y *
        (133 * x6 * z2 - 7 * x6 - 665 * x4 * y2 * z2 + 35 * x4 * y2 +
         399 * x2 * y4 * z2 - 21 * x2 * y4 - 19 * y6 * z2 + y6);
    dvalue_ddir.z += SH_C10[4] * sh[104] * 272 * xyz *
                     (57 * x4 * z2 - 9 * x4 - 190 * x2 * y2 * z2 +
                      30 * x2 * y2 + 57 * y4 * z2 - 9 * y4);
    dvalue_ddir.z += SH_C10[5] * sh[105] * 10 * y *
                     (1615 * x4 * z4 - 510 * x4 * z2 + 15 * x4 -
                      3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                      323 * y4 * z4 - 102 * y4 * z2 + 3 * y4);
    dvalue_ddir.z += SH_C10[6] * sh[106] * 48 * xyz *
                     (323 * x2 * z4 - 170 * x2 * z2 + 15 * x2 - 323 * y2 * z4 +
                      170 * y2 * z2 - 15 * y2);
    dvalue_ddir.z += SH_C10[7] * sh[107] * 14 * y *
                     (969 * x2 * z6 - 765 * x2 * z4 + 135 * x2 * z2 - 3 * x2 -
                      323 * y2 * z6 + 255 * y2 * z4 - 45 * y2 * z2 + y2);
    dvalue_ddir.z +=
        SH_C10[8] * sh[108] * 416 * xyz * (323 * z6 - 357 * z4 + 105 * z2 - 7);
    dvalue_ddir.z += SH_C10[9] * sh[109] * 18 * y *
                     (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dvalue_ddir.z +=
        SH_C10[10] * sh[110] * z *
        (461890 * z8 - 875160 * z6 + 540540 * z4 - 120120 * z2 + 6930);
    dvalue_ddir.z += SH_C10[9] * sh[111] * 18 * x *
                     (4199 * z8 - 6188 * z6 + 2730 * z4 - 364 * z2 + 7);
    dvalue_ddir.z += SH_C10[8] * sh[112] * 208 * z *
                     (323 * x2 * z6 - 357 * x2 * z4 + 105 * x2 * z2 - 7 * x2 -
                      323 * y2 * z6 + 357 * y2 * z4 - 105 * y2 * z2 + 7 * y2);
    dvalue_ddir.z += SH_C10[7] * sh[113] * 14 * x *
                     (323 * x2 * z6 - 255 * x2 * z4 + 45 * x2 * z2 - x2 -
                      969 * y2 * z6 + 765 * y2 * z4 - 135 * y2 * z2 + 3 * y2);
    dvalue_ddir.z += SH_C10[6] * sh[114] * 12 * z *
                     (323 * x4 * z4 - 170 * x4 * z2 + 15 * x4 -
                      1938 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 90 * x2 * y2 +
                      323 * y4 * z4 - 170 * y4 * z2 + 15 * y4);
    dvalue_ddir.z += SH_C10[5] * sh[115] * 10 * x *
                     (323 * x4 * z4 - 102 * x4 * z2 + 3 * x4 -
                      3230 * x2 * y2 * z4 + 1020 * x2 * y2 * z2 - 30 * x2 * y2 +
                      1615 * y4 * z4 - 510 * y4 * z2 + 15 * y4);
    dvalue_ddir.z +=
        SH_C10[4] * sh[116] * 136 * z *
        (19 * x6 * z2 - 3 * x6 - 285 * x4 * y2 * z2 + 45 * x4 * y2 +
         285 * x2 * y4 * z2 - 45 * x2 * y4 - 19 * y6 * z2 + 3 * y6);
    dvalue_ddir.z +=
        SH_C10[3] * sh[117] * 6 * x *
        (19 * x6 * z2 - x6 - 399 * x4 * y2 * z2 + 21 * x4 * y2 +
         665 * x2 * y4 * z2 - 35 * x2 * y4 - 133 * y6 * z2 + 7 * y6);
    dvalue_ddir.z += SH_C10[2] * sh[118] * 76 * z *
                     (x8 - 28 * x6 * y2 + 70 * x4 * y4 - 28 * x2 * y6 + y8);
    dvalue_ddir.z +=
        SH_C10[1] * sh[119] * 2 * x *
        (x8 - 36 * x6 * y2 + 126 * x4 * y4 - 84 * x2 * y6 + 9 * y8);

    dL_ddir[0].x += dvalue_ddir.x * dL_dval;
    dL_ddir[0].y += dvalue_ddir.y * dL_dval;
    dL_ddir[0].z += dvalue_ddir.z * dL_dval;
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
                                            const float *dL_dval,
                                            float *dL_dshs,
                                            float3 *dL_ddirs) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !visible[idx])
        return;

    // SH: [N, C, D]
    // dir: [N, 3]
    // dL_dval: [N, C]
    for (int i = 0; i < C; i++) {
        const float *dL_dval_tmp = dL_dval + idx * C;
        const float *sh_tmp = shs + idx * D * C + i * D;
        float *dL_dsh_tmp = dL_dshs + idx * D * C + i * D;
        float3 *dL_ddir_tmp = dL_ddirs + idx;

        evaluateSH0Backward(
            sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 1)
            evaluateSH1Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 4)
            evaluateSH2Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 9)
            evaluateSH3Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 16)
            evaluateSH4Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 25)
            evaluateSH5Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 36)
            evaluateSH6Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 49)
            evaluateSH7Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 64)
            evaluateSH8Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 81)
            evaluateSH9Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
        if (D > 100)
            evaluateSH10Backward(
                sh_tmp, dirs[idx], dL_dval_tmp[i], dL_dsh_tmp, dL_ddir_tmp);
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
            value.contiguous().data_ptr<float>());
    }

    return value;
}

std::tuple<torch::Tensor, torch::Tensor>
computeSHBackward(const torch::Tensor &shs,
                  const torch::Tensor &view_dirs,
                  const torch::Tensor &visible,
                  const torch::Tensor &dL_dval) {
    CHECK_INPUT(shs);
    CHECK_INPUT(view_dirs);
    CHECK_INPUT(visible);
    CHECK_INPUT(dL_dval);

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
            dL_dval.contiguous().data_ptr<float>(),
            dL_dshs.contiguous().data_ptr<float>(),
            (float3 *)dL_dvdirs.contiguous().data_ptr<float>());
    }

    return std::make_tuple(dL_dshs, dL_dvdirs);
}