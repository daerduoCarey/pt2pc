from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sampling_ext',
    ext_modules=[
        CUDAExtension('sampling_cuda', [
            'sampling_api.cpp',
            'sampling.cpp', 
            'sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
