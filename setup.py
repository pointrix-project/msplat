from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = [
	os.path.join(ROOT_DIR, "src", "ext.cpp"),
	os.path.join(ROOT_DIR, "src", "preprocess.cu"),
	os.path.join(ROOT_DIR, "src", "render.cu"),
	os.path.join(ROOT_DIR, "src", "sh.cu"),
]

def make_extension():
	ext = CUDAExtension(
		name=f"DifferentiablePointRender._C",
		sources=source_files,
		include_dirs=[
            os.path.join(ROOT_DIR, "include"),
            os.path.join(ROOT_DIR, "third_party", "glm"),
        ],
	)
	return ext

setup(
    name="DifferentiablePointRender",
	description="DifferentiablePointRender extension for Pointrix",
    url="https://github.com/NJU-3DV/DifferentiablePointRender",
    packages=['DifferentiablePointRender'],
    ext_modules=[make_extension()],
	cmdclass={"build_ext": BuildExtension}
)
