# coding=utf-8
r"""
PyCharm Editor
"""
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extra_compile_args = ["-std=c++17", "-march=native", ] #"-ffast-math"
language = "c++"
# link to gcc headers may be required if not find automatically
# eg on ARM with gcc 13 here is a fix for the linker
#extra_link_args = ['-Wl,-rpath,/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/13']
language = "c++"

setup(name="cpp",
      ext_modules=cythonize([
          Extension(
              name="cpp.bandits",
              sources=["cpp/bandits.pyx",
                       "cpp/src/utils.cxx"],
              language=language,
              extra_compile_args=extra_compile_args,
              #extra_link_args=extra_link_args
          ),
          Extension(
             name="cpp.policies",
             sources=["cpp/policies.pyx",
                      "cpp/src/utils.cxx"],
             language=language,
             extra_compile_args=extra_compile_args,
             #extra_link_args=extra_link_args
         )
      ]),
      language_level=3,
      include_dirs=[np.get_include()]
      )
