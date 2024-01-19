
#include <compute_cov3d.h>
#include <project_point.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_cov3d_forward", &computeCov3DForward);
  m.def("compute_cov3d_backward", &computeCov3DBackward);
  m.def("project_point_forward", &projectPointsForward);
  m.def("project_point_backward", &projectPointsBackward);
}
