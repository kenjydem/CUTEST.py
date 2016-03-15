from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import ConfigParser
import subprocess
import os
import sys
import numpy


cutest_config = ConfigParser.SafeConfigParser()
cutest_config.read('site.cfg')

if os.getenv('CUTESTLIB') is None:
    raise NameError('You have to set an environment variable CUTESTLIB to specify where to register your problems')
    sys.exit([1])

default_library_dir = cutest_config.get('DEFAULT', 'library_dirs').split(os.pathsep)
default_include_dir = cutest_config.get('DEFAULT', 'include_dirs').split(os.pathsep)

ext_modules = [Extension(name="cutest.ccutest",
                         sources = [os.path.join("cutest", "ccutest.pyx")],
                         library_dirs = default_library_dir, 
			 include_dirs = [numpy.get_include()],
                         libraries = ['cutest'])] 

setup(
    name = 'CUTEst.py',
    version='0.0.1',
    description='Python interface for CUTEst library',
    author=['Kenjy Demeester','Farooq Sanni'],
    ext_modules = ext_modules,
    cmdclass={'build_ext': build_ext},
    package_dir={"cutest": "cutest"},
    packages=["cutest"],
    )
