from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print(ROOT_DIR)

source_files = [
	os.path.join(ROOT_DIR, "src", "cov.cu"),
	os.path.join(ROOT_DIR, "src", "preprocess.cu"),
	os.path.join(ROOT_DIR, "src", "render.cu"),
	os.path.join(ROOT_DIR, "src", "sh.cu"),
	os.path.join(ROOT_DIR, "src", "ext.cpp"),
]

def make_extension():
	ext = CUDAExtension(
		name=f"GS_Split._C",
		sources=source_files,
		include_dirs=[
            os.path.join(ROOT_DIR, "include"),
            os.path.join(ROOT_DIR, "third_party", "glm"),
        ],
		extra_compile_args={"nvcc": ["-I" + os.path.join(ROOT_DIR, "third_party/glm/")]}
	)
	return ext

setup(
    name="GS_Split",
    description="DifferentiablePointRender extension for Pointrix",
    url="https://github.com/NJU-3DV/DifferentiablePointRender",
    packages=['.'],
    ext_modules=[make_extension()],
    cmdclass={"build_ext": BuildExtension}
)
