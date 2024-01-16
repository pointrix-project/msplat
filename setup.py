
import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__version__ = "1.0.0"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)

def search_sources():
    renderer_names = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(d)]

    sources = []
    for renderer in renderer_names:
        sources += glob.glob(os.path.join(ROOT_DIR, renderer, "src/*.c"))
        sources += glob.glob(os.path.join(ROOT_DIR, renderer, "src/*.cu")) 
        sources += glob.glob(os.path.join(ROOT_DIR, renderer, "src/*.cpp"))

    return sources


def search_includes():
    renderer_names = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(d)]

    includes = []
    for renderer in renderer_names:
        includes += glob.glob(os.path.join(ROOT_DIR, renderer, "include"))

    return includes


def search_third_party():
    third_party_dir = os.path.join(ROOT_DIR, "third_party")
    
    third_parties = []
    if os.path.exists(third_party_dir):
        third_parties = [os.path.join(third_party_dir, d) 
            for d in os.listdir(third_party_dir)]

    return third_parties
    

def make_extension():
    
	ext = CUDAExtension(
		name=f"DifferentiablePointRender._C",
		sources=search_sources(),
		include_dirs=search_includes()+search_third_party(),
		extra_compile_args={"nvcc": ["-O3", "--use_fast_math"]}
	)
	return ext


setup(
    name="DifferentiablePointRender",
    version=__version__,
	description="Differentiable Point Render CUDA extension for Pointrix",
    url="https://github.com/NJU-3DV/DifferentiablePointRender",
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "jaxtyping"
    ],
    packages=['DifferentiablePointRender'],
    ext_modules=[make_extension()],
	cmdclass={"build_ext": BuildExtension}
)
