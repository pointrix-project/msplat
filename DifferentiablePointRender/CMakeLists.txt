cmake_minimum_required(VERSION 3.20)

project(DPR LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(DifferentiablePointRender STATIC
	GaussianSplatting/src/compute_cov3d.cu
	GaussianSplatting/src/compute_cov3d.cpp
	GaussianSplatting/src/ext.cpp
)

set_target_properties(DifferentiablePointRender PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(DifferentiablePointRender PUBLIC 
	"GaussianSplatting/include"
)

target_include_directories(DifferentiablePointRender PRIVATE 
	"third_party/glm" 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)