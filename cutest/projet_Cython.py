import os
import platform
import subprocess
from  cutest.ccutest import  loadCutestProb


CUTEST_ARCH = os.getenv('MYARCH')
CUTEST_DIR = os.getenv('CUTEST')
OUTSDIF = "OUTSDIF.d"
AUTOMAT = "AUTOMAT.d"


if platform.system() == "Darwin":
    linker = "gfortran"
    sh_flags = "-shared"
    soname = "dylib"
else:
    raise NotImplementedError

# Decode problem and build shared library.
def sifdecoder(name):
  pname, sif = name.split('.')
  libname = 'lib'+ pname
  subprocess.call(['sifdecoder', name])
  subprocess.call('gfortran -c -fPIC ELFUN.f EXTER.f GROUP.f RANGE.f',shell=True)
  subprocess.call([linker,sh_flags,"-o",libname+"."+ soname,"ELFUN.o","EXTER.o","GROUP.o","RANGE.o","-L"+ CUTEST_DIR+os.path.sep+"objects"+os.path.sep+CUTEST_ARCH+os.path.sep+"double"])
  subprocess.call('rm ELFUN.f EXTER.f GROUP.f RANGE.f ELFUN.o EXTER.o GROUP.o RANGE.o',shell=True)
 if os.path.isfile("OUTSDIF.d") == False:
 	raise AssertionError("File OUTSDIF.d not exist")
 return libname



if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    sifdecoder(name)
    loadCutestProb(name)
