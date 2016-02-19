from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [Extension(name="cutest.ccutest",
                         sources=[os.path.join("cutest", "ccutest.pyx")],
                         library_dirs=['/usr/local/lib/','/usr/local/lib/gcc/5/', '/usr/lib'],
			 libraries=['gsl', 'cutest'])]

setup(
    name = "CUTEst.py",
    ext_modules = ext_modules,
    cmdclass={'build_ext': build_ext},
    package_dir={"cutest": "cutest"},
    packages=["cutest"],
    zip_safe=False
)
