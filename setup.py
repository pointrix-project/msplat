
import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__version__ = "1.0.0"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def search_sources():
    
    sources = glob.glob(os.path.join(f"{ROOT_DIR}", "msplat/src/*.c"))
    sources += glob.glob(os.path.join(f"{ROOT_DIR}", "msplat/src/*.cu")) 
    sources += glob.glob(os.path.join(f"{ROOT_DIR}", "msplat/src/*.cpp"))

    return sources


def search_third_party():
    third_party_dir = os.path.join(ROOT_DIR, "third_party")
    
    third_parties = []
    if os.path.exists(third_party_dir):
        third_parties = [os.path.join(third_party_dir, d) 
            for d in os.listdir(third_party_dir)]

    return third_parties
    

def make_extensions():
    # -g for debug information
    # -O3 for higher level of optimization
    # --use_fast_math for results in a faster runtime, but may sacrifice some mathematical precision.
    exts = [CUDAExtension(
            name="msplat._C",
            sources=search_sources(),
            include_dirs= [os.path.join(f"{ROOT_DIR}", "msplat/include")] + search_third_party(),
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ]
    
    return exts

setup(
    name="msplat",
    version=__version__,
    description="a modular differential gaussian rasterization library",
    url="https://github.com/pointrix-project/msplat",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "jaxtyping"
    ],
    ext_modules=make_extensions(),
    cmdclass={"build_ext": BuildExtension}
)
