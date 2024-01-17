
#include <compute_cov3d.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_cov3d_forward", &computeCov3DForward);
  m.def("compute_cov3d_backward", &computeCov3DBackward);
}
