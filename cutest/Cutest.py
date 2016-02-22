import os
import sys
import platform
import subprocess
try:
    from  cutest.ccutest import  loadCutestProb
except ImportError:
    print("Dynamic library problem unfound, please compiled with creatlib.py")
    sys.exit(0)


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
	
if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    loadCutestProb(name)
