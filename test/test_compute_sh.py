
import time
import torch
import numpy as np
import msplat as ms
from scipy.special import sph_harm

def compute_sh_torch_impl(sh_coeffs, viewdirs):
    *dims, C, dim_sh = sh_coeffs.shape
    
    bases = eval_sh_bases(dim_sh, viewdirs)
    
    return (bases[:, None, :] * sh_coeffs).sum(dim=-1)


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
SH_C5 = [
    -0.3281910284200851,
    1.0378311574405208,
    -0.2446191497176252, 
    1.198384196243331, 
    -0.22647332559784847, 
    0.1169503224534236, 
    -0.22647332559784847, 
    1.198384196243331,
    -0.2446191497176252,
    1.0378311574405208,
    -0.3281910284200851
]

SH_C6 = [
    0.34159205259595715, 
    -1.183309581115876, 
    0.2522824503643621, 
    -0.46060262975746175, 
    0.23030131487873087, 
    -0.2913106812593657, 
    0.06356920226762842, 
    -0.2913106812593657, 
    0.23030131487873087, 
    -0.46060262975746175, 
    0.2522824503643621, 
    -1.183309581115876, 
    0.34159205259595715
]

SH_C7 = [
    -0.3535813662622981, 
    1.32298033090095, 
    -0.2594577893601302, 
    0.5189155787202604, 
    -0.07822946693114702, 
    0.11063317311124565, 
    -0.04516580379125865, 
    0.06828427691200495, 
    -0.04516580379125865, 
    0.11063317311124565, 
    -0.07822946693114702, 
    0.5189155787202604,
    -0.2594577893601302,
    1.32298033090095,
    -0.3535813662622981
]

SH_C8 = [
    0.36446333008741494,
    -1.4578533203496598, 
    0.2661663830297713, 
    -1.724955311049054,
    0.23920826237966533,
    -0.6176330776477721,
    0.22807612921745474,
    -0.054520622949389974,
    0.009086770491564996,
    -0.054520622949389974,
    0.22807612921745474,
    -0.6176330776477721, 
    0.23920826237966533,
    -1.724955311049054,
    0.2661663830297713,
    -1.4578533203496598,
    0.36446333008741494
]

SH_C9 = [
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

SH_C10 = [
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

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    
    result[..., 0] = SH_C0
    
    x, y, z = dirs.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    xxx, yyy, zzz, xyz = x * x * x, y * y * y, z * z * z, x * y * z
    
    if basis_dim > 1:
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
    
    if basis_dim > 4:
        result[..., 4] = SH_C2[0] * xy
        result[..., 5] = SH_C2[1] * yz
        result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
        result[..., 7] = SH_C2[3] * xz
        result[..., 8] = SH_C2[4] * (xx - yy)

    if basis_dim > 9:
        result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
        result[..., 10] = SH_C3[1] * xy * z
        result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
        result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
        result[..., 14] = SH_C3[5] * z * (xx - yy)
        result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

    if basis_dim > 16:
        result[..., 16] = SH_C4[0] * xy * (xx - yy)
        result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
        result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
        result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
        result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
        result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
        result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
        result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
        result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                    
    if basis_dim > 25:
        result[..., 25] = SH_C5[0] * (10 * xxx * xy - 20 * xx * yyy + 2 * yyy * yy)
        result[..., 26] = SH_C5[1] * (8 * xxx * yz - 8 * xy * yy * z)
        result[..., 27] = SH_C5[2] * (54 * xx * y * zz - 6 * xx * y - 18 * yyy * zz + 2 * yyy)
        result[..., 28] = SH_C5[3] * (12 * xy * zzz - 4 * x * yz)
        result[..., 29] = SH_C5[4] * (42 * yz * zzz - 28 * y * zz + 2 * y)
        result[..., 30] = SH_C5[5] * (63 * zz * zzz - 70 * zzz + 15 * z)
        result[..., 31] = SH_C5[6] * (42 * x * zz * zz - 28 * x * zz + 2 * x)
        result[..., 32] = SH_C5[7] * (6 * xx * zzz - 2 * xx * z - 6 * yy * zzz + 2 * yy * z)
        result[..., 33] = SH_C5[8] * (18 * xxx * zz - 2 * xxx - 54 * x * yy * zz + 6 * x * yy)
        result[..., 34] = SH_C5[9] * (2 * xx * xx * z - 12 * xx * yy * z + 2 * yy * yy * z)
        result[..., 35] = SH_C5[10] * (2 * xxx * xx - 20 * xxx * yy + 10 * xy * yyy)
    
    if basis_dim > 36:
        result[..., 36] = SH_C6[0] * (12 * xxx * xx * y - 40 * xxx * yyy + 12 * xy * yy * yy)
        result[..., 37] = SH_C6[1] * (10 * xx * xx * yz - 20 * xx * yy * yz + 2 * yy * yy * yz)
        result[..., 38] = SH_C6[2] * (88 * xx * xy * zz - 8 * xx * xy - 88 * xy * yy * zz + 8 * xy * yy)
        result[..., 39] = SH_C6[3] * 2 * yz * (33 * xx * zz - 9 * xx - 11 * yy * zz + 3 * yy)
        result[..., 40] = SH_C6[4] * 4 * xy * (33 * zz * zz - 18 * zz + 1)
        result[..., 41] = SH_C6[5] * 2 * yz * (33 * zz * zz - 30 * zz + 5)
        result[..., 42] = SH_C6[6] * (231 * zzz * zzz - 315 * zz * zz + 105 * zz - 5)
        result[..., 43] = SH_C6[7] * (66 * xz * zz * zz - 60 * xz * zz  + 10 * xz)
        result[..., 44] = SH_C6[8] * (66 * xx * zz * zz - 36 * xx * zz + 2 * xx - 66 * yy * zz * zz + 36 * yy * zz - 2 * yy)
        result[..., 45] = SH_C6[9] * (22 * xxx * zzz - 6 * xx * xz - 66 * xy * yz * zz + 18 * xy * yz)
        result[..., 46] = SH_C6[10] * (22 * xx * xx * zz - 2 * xx * xx - 132 * xx * yy * zz + 12 * xx * yy + 22 * yy * yy * zz - 2 * yy * yy)
        result[..., 47] = SH_C6[11] * (2 * xx * xx * xz - 20 * xx * xy * yz + 10 * xy * yy * yz)
        result[..., 48] = SH_C6[12] * (2 * xx * xx * xx - 30 * xx * xx * yy + 30 * xx * yy * yy - 2* yyy * yyy)
    
    if basis_dim > 49:
        result[..., 49] = SH_C7[0] * 2 * y * ( 7 * xxx * xxx - 35 * xx * xx * yy + 21 * xx * yy * yy - yyy * yyy)
        result[..., 50] = SH_C7[1] * 4 * xyz * (3 * xx * xx - 10 * xx * yy + 3 * yy * yy)
        result[..., 51] = SH_C7[2] * 2 * y * (65 * xx * xx * zz - 5* xx * xx - 130 * xx * yy * zz + 10 * xx * yy + 13 * yy * yy * zz - yy * yy)
        result[..., 52] = SH_C7[3] * 8 * xyz *(13 * xx * zz - 3 * xx - 13 * yy * zz + 3 * yy)
        result[..., 53] = SH_C7[4] * 2 * y * (429 * xx * zz * zz - 198 * xx * zz + 9 * xx - 143 * yy * zz * zz + 66 * yy * zz - 3 * yy)
        result[..., 54] = SH_C7[5] * 4 * xyz * (143 * zz * zz - 110 * zz + 15)
        result[..., 55] = SH_C7[6] * 2 * y * (429 * zzz * zzz - 495 * zz * zz + 135 * zz - 5)
        result[..., 56] = SH_C7[7] * z * ( 429 * zzz * zzz - 693 * zz * zz + 315 * zz - 35)
        result[..., 57] = SH_C7[8] * 2 * x * (429 * zzz * zzz - 495 * zz * zz + 135 * zz - 5)
        result[..., 58] = SH_C7[9] * 2 * z * (143 * xx * zz * zz - 110 * xx * zz + 15 * xx - 143 * yy * zz * zz + 110 * yy * zz - 15 * yy)
        result[..., 59] = SH_C7[10] * 2 * x * (143 * xx * zz * zz - 66 * xx * zz + 3 * xx - 429 * yy * zz * zz + 198 * yy * zz - 9 * yy)
        result[..., 60] = SH_C7[11] * 2 * z * (13 * xx * xx * zz - 3 * xx * xx - 78 * xx * yy * zz + 18 * xx * yy + 13 * yy * yy * zz - 3 * yy * yy)
        result[..., 61] = SH_C7[12] * 2 * x * (13 * xx * xx * zz - xx * xx - 130 * xx * yy * zz + 10 * xx * yy + 65 * yy * yy * zz - 5 * yy * yy)
        result[..., 62] = SH_C7[13] * 2 * z * (xxx * xxx - 15 * xx * xx * yy + 15 * xx * yy * yy - yyy * yyy)
        result[..., 63] = SH_C7[14] * 2 * x * (xxx * xxx - 21 * xx * xx * yy + 35 * xx * yy * yy - 7 * yyy * yyy)
    
    if basis_dim > 64:
        result[..., 64] = SH_C8[0] * 16*x*y*(x**6 - 7*x**4*y**2 + 7*x**2*y**4 - y**6)
        result[..., 65] = SH_C8[1] * 2*y*z*(7*x**6 - 35*x**4*y**2 + 21*x**2*y**4 - y**6)
        result[..., 66] = SH_C8[2] * 4*x*y*(45*x**4*z**2 - 3*x**4 - 150*x**2*y**2*z**2 + 10*x**2*y**2 + 45*y**4*z**2 - 3*y**4)
        result[..., 67] = SH_C8[3] * 2*y*z*(25*x**4*z**2 - 5*x**4 - 50*x**2*y**2*z**2 + 10*x**2*y**2 + 5*y**4*z**2 - y**4)
        result[..., 68] = SH_C8[4] * 8*x*y*(65*x**2*z**4 - 26*x**2*z**2 + x**2 - 65*y**2*z**4 + 26*y**2*z**2 - y**2)
        result[..., 69] = SH_C8[5] * 2*y*z*(117*x**2*z**4 - 78*x**2*z**2 + 9*x**2 - 39*y**2*z**4 + 26*y**2*z**2 - 3*y**2)
        result[..., 70] = SH_C8[6] * 4*x*y*(143*z**6 - 143*z**4 + 33*z**2 - 1)
        result[..., 71] = SH_C8[7] * 2*y*z*(715*z**6 - 1001*z**4 + 385*z**2 - 35)
        result[..., 72] = SH_C8[8] * (6435*z**8 - 12012*z**6 + 6930*z**4 - 1260*z**2 + 35)
        result[..., 73] = SH_C8[9] * 2*x*z*(715*z**6 - 1001*z**4 + 385*z**2 - 35)
        result[..., 74] = SH_C8[10] * (286*x**2*z**6 - 286*x**2*z**4 + 66*x**2*z**2 - 2*x**2 - 286*y**2*z**6 + 286*y**2*z**4 - 66*y**2*z**2 + 2*y**2)
        result[..., 75] = SH_C8[11] * 2*x*z*(39*x**2*z**4 - 26*x**2*z**2 + 3*x**2 - 117*y**2*z**4 + 78*y**2*z**2 - 9*y**2)
        result[..., 76] = SH_C8[12] * (130*x**4*z**4 - 52*x**4*z**2 + 2*x**4 - 780*x**2*y**2*z**4 + 312*x**2*y**2*z**2 - 12*x**2*y**2 + 130*y**4*z**4 - 52*y**4*z**2 + 2*y**4)
        result[..., 77] = SH_C8[13] * 2*x*z*(5*x**4*z**2 - x**4 - 50*x**2*y**2*z**2 + 10*x**2*y**2 + 25*y**4*z**2 - 5*y**4)
        result[..., 78] = SH_C8[14] * (30*x**6*z**2 - 2*x**6 - 450*x**4*y**2*z**2 + 30*x**4*y**2 + 450*x**2*y**4*z**2 - 30*x**2*y**4 - 30*y**6*z**2 + 2*y**6)
        result[..., 79] = SH_C8[15] * 2*x*z*(x**6 - 21*x**4*y**2 + 35*x**2*y**4 - 7*y**6)
        result[..., 80] = SH_C8[16] * (2*x**8 - 56*x**6*y**2 + 140*x**4*y**4 - 56*x**2*y**6 + 2*y**8)
    
    if basis_dim > 81:
        result[..., 81] = SH_C9[0] * 2*y*(9*x**8 - 84*x**6*y**2 + 126*x**4*y**4 - 36*x**2*y**6 + y**8)
        result[..., 82] = SH_C9[1] *16*x*y*z*(x**6 - 7*x**4*y**2 + 7*x**2*y**4 - y**6)
        result[..., 83] = SH_C9[2] *2*y*(119*x**6*z**2 - 7*x**6 - 595*x**4*y**2*z**2 + 35*x**4*y**2 + 357*x**2*y**4*z**2 - 21*x**2*y**4 - 17*y**6*z**2 + y**6)
        result[..., 84] = SH_C9[3] *4*x*y*z*(51*x**4*z**2 - 9*x**4 - 170*x**2*y**2*z**2 + 30*x**2*y**2 + 51*y**4*z**2 - 9*y**4)
        result[..., 85] = SH_C9[4] *2*y*(425*x**4*z**4 - 150*x**4*z**2 + 5*x**4 - 850*x**2*y**2*z**4 + 300*x**2*y**2*z**2 - 10*x**2*y**2 + 85*y**4*z**4 - 30*y**4*z**2 + y**4)
        result[..., 86] = SH_C9[5] *8*x*y*z*(17*x**2*z**4 - 10*x**2*z**2 + x**2 - 17*y**2*z**4 + 10*y**2*z**2 - y**2)
        result[..., 87] = SH_C9[6] *2*y*(663*x**2*z**6 - 585*x**2*z**4 + 117*x**2*z**2 - 3*x**2 - 221*y**2*z**6 + 195*y**2*z**4 - 39*y**2*z**2 + y**2)
        result[..., 88] = SH_C9[7] *4*x*y*z*(221*z**6 - 273*z**4 + 91*z**2 - 7)
        result[..., 89] = SH_C9[8] *2*y*(2431*z**8 - 4004*z**6 + 2002*z**4 - 308*z**2 + 7)
        result[..., 90] = SH_C9[9] * z * (12155*z**8 - 25740*z**6 + 18018*z**4 - 4620*z**2 + 315)
        result[..., 91] = SH_C9[10] *2*x*(2431*z**8 - 4004*z**6 + 2002*z**4 - 308*z**2 + 7)
        result[..., 92] = SH_C9[11] *2*z*(221*x**2*z**6 - 273*x**2*z**4 + 91*x**2*z**2 - 7*x**2 - 221*y**2*z**6 + 273*y**2*z**4 - 91*y**2*z**2 + 7*y**2)
        result[..., 93] = SH_C9[12] *2*x*(221*x**2*z**6 - 195*x**2*z**4 + 39*x**2*z**2 - x**2 - 663*y**2*z**6 + 585*y**2*z**4 - 117*y**2*z**2 + 3*y**2)
        result[..., 94] = SH_C9[13] *2*z*(17*x**4*z**4 - 10*x**4*z**2 + x**4 - 102*x**2*y**2*z**4 + 60*x**2*y**2*z**2 - 6*x**2*y**2 + 17*y**4*z**4 - 10*y**4*z**2 + y**4)
        result[..., 95] = SH_C9[14] *2*x*(85*x**4*z**4 - 30*x**4*z**2 + x**4 - 850*x**2*y**2*z**4 + 300*x**2*y**2*z**2 - 10*x**2*y**2 + 425*y**4*z**4 - 150*y**4*z**2 + 5*y**4)
        result[..., 96] = SH_C9[15] *2*z*(17*x**6*z**2 - 3*x**6 - 255*x**4*y**2*z**2 + 45*x**4*y**2 + 255*x**2*y**4*z**2 - 45*x**2*y**4 - 17*y**6*z**2 + 3*y**6)
        result[..., 97] = SH_C9[16] *2*x*(17*x**6*z**2 - x**6 - 357*x**4*y**2*z**2 + 21*x**4*y**2 + 595*x**2*y**4*z**2 - 35*x**2*y**4 - 119*y**6*z**2 + 7*y**6)
        result[..., 98] = SH_C9[17] *2*z*(x**8 - 28*x**6*y**2 + 70*x**4*y**4 - 28*x**2*y**6 + y**8)
        result[..., 99] = SH_C9[18] *2*x*(x**8 - 36*x**6*y**2 + 126*x**4*y**4 - 84*x**2*y**6 + 9*y**8)
    
    if basis_dim > 100:
        result[..., 100] = SH_C10[0] * (20*x**9*y - 240*x**7*y**3 + 504*x**5*y**5 - 240*x**3*y**7 + 20*x*y**9)
        result[..., 101] = SH_C10[1] *2*y*z*(9*x**8 - 84*x**6*y**2 + 126*x**4*y**4 - 36*x**2*y**6 + y**8)
        result[..., 102] = SH_C10[2] *16*x*y*(19*x**6*z**2 - x**6 - 133*x**4*y**2*z**2 + 7*x**4*y**2 + 133*x**2*y**4*z**2 - 7*x**2*y**4 - 19*y**6*z**2 + y**6)
        result[..., 103] = SH_C10[3] *2*y*z*(133*x**6*z**2 - 21*x**6 - 665*x**4*y**2*z**2 + 105*x**4*y**2 + 399*x**2*y**4*z**2 - 63*x**2*y**4 - 19*y**6*z**2 + 3*y**6)
        result[..., 104] = SH_C10[4] *4*x*y*(969*x**4*z**4 - 306*x**4*z**2 + 9*x**4 - 3230*x**2*y**2*z**4 + 1020*x**2*y**2*z**2 - 30*x**2*y**2 + 969*y**4*z**4 - 306*y**4*z**2 + 9*y**4)
        result[..., 105] = SH_C10[5] *2*y*z*(1615*x**4*z**4 - 850*x**4*z**2 + 75*x**4 - 3230*x**2*y**2*z**4 + 1700*x**2*y**2*z**2 - 150*x**2*y**2 + 323*y**4*z**4 - 170*y**4*z**2 + 15*y**4)
        result[..., 106] = SH_C10[6] *8*x*y*(323*x**2*z**6 - 255*x**2*z**4 + 45*x**2*z**2 - x**2 - 323*y**2*z**6 + 255*y**2*z**4 - 45*y**2*z**2 + y**2)
        result[..., 107] = SH_C10[7] *2*y*z*(969*x**2*z**6 - 1071*x**2*z**4 + 315*x**2*z**2 - 21*x**2 - 323*y**2*z**6 + 357*y**2*z**4 - 105*y**2*z**2 + 7*y**2)
        result[..., 108] = SH_C10[8] *4*x*y*(4199*z**8 - 6188*z**6 + 2730*z**4 - 364*z**2 + 7)
        result[..., 109] = SH_C10[9] *2*y*z*(4199*z**8 - 7956*z**6 + 4914*z**4 - 1092*z**2 + 63)
        result[..., 110] = SH_C10[10] *(46189*z**10 - 109395*z**8 + 90090*z**6 - 30030*z**4 + 3465*z**2 - 63)
        result[..., 111] = SH_C10[11] *2*x*z*(4199*z**8 - 7956*z**6 + 4914*z**4 - 1092*z**2 + 63)
        result[..., 112] = SH_C10[12] *(8398*x**2*z**8 - 12376*x**2*z**6 + 5460*x**2*z**4 - 728*x**2*z**2 + 14*x**2 - 8398*y**2*z**8 + 12376*y**2*z**6 - 5460*y**2*z**4 + 728*y**2*z**2 - 14*y**2)
        result[..., 113] = SH_C10[13] *2*x*z*(323*x**2*z**6 - 357*x**2*z**4 + 105*x**2*z**2 - 7*x**2 - 969*y**2*z**6 + 1071*y**2*z**4 - 315*y**2*z**2 + 21*y**2)
        result[..., 114] = SH_C10[14] *(646*x**4*z**6 - 510*x**4*z**4 + 90*x**4*z**2 - 2*x**4 - 3876*x**2*y**2*z**6 + 3060*x**2*y**2*z**4 - 540*x**2*y**2*z**2 + 12*x**2*y**2 + 646*y**4*z**6 - 510*y**4*z**4 + 90*y**4*z**2 - 2*y**4)
        result[..., 115] = SH_C10[15] *2*x*z*(323*x**4*z**4 - 170*x**4*z**2 + 15*x**4 - 3230*x**2*y**2*z**4 + 1700*x**2*y**2*z**2 - 150*x**2*y**2 + 1615*y**4*z**4 - 850*y**4*z**2 + 75*y**4)
        result[..., 116] = SH_C10[16] *(646*x**6*z**4 - 204*x**6*z**2 + 6*x**6 - 9690*x**4*y**2*z**4 + 3060*x**4*y**2*z**2 - 90*x**4*y**2 + 9690*x**2*y**4*z**4 - 3060*x**2*y**4*z**2 + 90*x**2*y**4 - 646*y**6*z**4 + 204*y**6*z**2 - 6*y**6)
        result[..., 117] = SH_C10[17] *2*x*z*(19*x**6*z**2 - 3*x**6 - 399*x**4*y**2*z**2 + 63*x**4*y**2 + 665*x**2*y**4*z**2 - 105*x**2*y**4 - 133*y**6*z**2 + 21*y**6)
        result[..., 118] = SH_C10[18] *(38*x**8*z**2 - 2*x**8 - 1064*x**6*y**2*z**2 + 56*x**6*y**2 + 2660*x**4*y**4*z**2 - 140*x**4*y**4 - 1064*x**2*y**6*z**2 + 56*x**2*y**6 + 38*y**8*z**2 - 2*y**8)
        result[..., 119] = SH_C10[19] *2*x*z*(x**8 - 36*x**6*y**2 + 126*x**4*y**4 - 84*x**2*y**6 + 9*y**8)
        result[..., 120] = SH_C10[20] *(2*x**10 - 90*x**8*y**2 + 420*x**6*y**4 - 420*x**4*y**6 + 90*x**2*y**8 - 2*y**10)
    
    return result


def eval_scipy(dirs: torch.Tensor, sh_coeffs):
    *dims, C, dim_sh = sh_coeffs.shape
    
    # only for N = 1
    result = torch.empty(
        (*dirs.shape[:-1], dim_sh), dtype=dirs.dtype, device=dirs.device
    )
    
    phi = torch.acos(dirs[..., 2])
    theta = torch.atan2(dirs[..., 1], dirs[..., 0])
    
    level = int(np.sqrt(dim_sh)) - 1
    for n in range(0, level + 1):
        for m in range(- n, n+1):
            y_mn1 = sph_harm(m, n, theta[0].cpu().numpy(), phi[0].cpu().numpy())
            y_mn2 = sph_harm(-m, n, theta[0].cpu().numpy(), phi[0].cpu().numpy())
            
            if m < 0:
                tmp = complex(0, np.sqrt(1/2))
                result[:, n * n + m + n] = np.power(-1, -m) * (tmp * (y_mn1 - np.power(-1, -m) * y_mn2)).real
                
            elif m > 0:
                tmp = complex(np.sqrt(1/2), 0)
                result[:, n * n + m + n] = np.power(-1, m) * (tmp * (y_mn2 + np.power(-1, m) * y_mn1)).real
            else:
                result[:, n * n + m + n] = y_mn1
            
    return (result[:, None, :] * sh_coeffs).sum(dim=-1)


if __name__ == "__main__":
    
    seed = 123
    torch.manual_seed(seed)
    
    # Only Forward: test all elvels with pytorch implement and scipy implement
    print("=============================== Verify pytorch implement for SH ===============================")
    support_level = 10
    for deg in range(support_level+1):
        degree_dim = pow(deg+1, 2)
        viewdirs = torch.randn(1, 3).cuda()
        viewdirs /= torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
        sh_coeffs = torch.randn(1, 1, degree_dim).cuda()
        
        out_scipy = eval_scipy(viewdirs, sh_coeffs)
        out_pytorch = compute_sh_torch_impl(sh_coeffs, viewdirs)
        
        torch.testing.assert_close(out_scipy, out_pytorch)
        print("Level ", deg, "Pytorch implement verified.")

    # Forward and backward: test on cuda
    iters = 10
    N = 1000
    degree = 10
    
    print("=============================== running test on compute_sh ===============================")
    # generate data
    degree_dim = pow(degree+1, 2)
    
    viewdirs = torch.randn(N, 3).cuda()
    viewdirs /= torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    sh_coeffs = torch.randn(N, 1, degree_dim).cuda()
    
    sh_coeffs1 = sh_coeffs.clone().requires_grad_()
    sh_coeffs2 = sh_coeffs.clone().requires_grad_()
    viewdirs1 = viewdirs.clone().requires_grad_()
    viewdirs2 = viewdirs.clone().requires_grad_()
    
    # ============================================ Forward =====================================
    print("forward: ")
    t = time.time()
    for i in range(iters):
        out_pytorch = compute_sh_torch_impl(sh_coeffs1, viewdirs1)

    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    for i in range(iters):
        out_cuda = ms.compute_sh(sh_coeffs2, viewdirs2)

    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(out_pytorch, out_cuda, atol=5e-4, rtol=1e-5)
    print("Forward pass.")
    
    # ============================================ Backward =====================================
    print("backward: ")
    t = time.time()
    loss = out_pytorch.mean()
    loss.backward()
    torch.cuda.synchronize()
    print("  pytorch runtime: ", (time.time() - t) / iters, " s")

    t = time.time()
    loss2 = out_cuda.mean()
    loss2.backward()
    torch.cuda.synchronize()
    print("  cuda runtime: ", (time.time() - t) / iters, " s")
    
    torch.testing.assert_close(sh_coeffs1.grad, sh_coeffs2.grad, atol=5e-4, rtol=1e-5)
    torch.testing.assert_close(viewdirs1.grad, viewdirs2.grad, atol=5e-4, rtol=1e-5)
    print("Backward pass.")
