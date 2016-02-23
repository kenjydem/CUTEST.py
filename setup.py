from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from cutest.creatlib import compile_sif_problem
import os
import sys

dirlibs=[]
libnames=[]
print len(sys.argv)

if len(sys.argv) == 1:
    raise 'You have to specify a problem name' 
else:
    while len(sys.argv)!=2:
        pb=sys.argv.pop(2)
        (tmpdir, libname)=compile_sif_problem(pb)
        dirlibs.append(tmpdir)
        libnames.append(libname[3:-6])

dirlibs.append('/usr/local/lib/')
libnames.append('cutest')
print dirlibs, libnames

ext_modules = [Extension(name="cutest.ccutest",
                         sources=[os.path.join("cutest", "ccutest.pyx")],
                         library_dirs=dirlibs,      #'/usr/local/lib/','/usr/lib'],
			 libraries=libnames)]       # 'cutest'])]

setup(
    name = 'CUTEst.py',
    version='1.0',
    description='Python interface for CUTEst library',
    author=['Kenjy Demeester','Farooq Sanni'],
    ext_modules = ext_modules,
    cmdclass={'build_ext': build_ext},
    package_dir={"cutest": "cutest"},
    packages=["cutest"],
    force=1,
    zip_safe=False
)
