import os
import sys
import subprocess
from sys import platform

CUTEST_ARCH = os.getenv('MYARCH')
CUTEST_DIR = os.getenv('CUTEST')
if CUTEST_DIR is None or CUTEST_ARCH is None:
    raise(KeyError, "Please check that CUTEST and MYARCH are set")
cutest_libdir_double = os.path.join(CUTEST_DIR, "objects", CUTEST_ARCH, "double")

fcompiler = "gfortran"
if platform == "linux" or platform == "linux2":
    linker = "ld"
    sh_flags = "-shared"
    soname = "so"
elif platform == "darwin":
    linker = "gfortran"
    sh_flags = ["-dynamiclib", "-undefined", "dynamic_lookup"]
    soname = "dylib"
elif platform == "win32":
    raise(ValueError, "Windows platforms are currently not supported")

problem = sys.argv[1]

# Decode problem and build shared library.
if problem[-4:] == ".SIF":
    problem = problem[:-4]
subprocess.call(['sifdecoder', problem])
libname = "lib%s.%s" % (problem, soname)

srcs = ["ELFUN", "RANGE", "GROUP", "EXTER"]
dat = ["OUTSDIF.d", "AUTOMAT.d"]

# Compile source files.
exit_code = subprocess.call([fcompiler, "-c"] +  [src + ".f" for src in srcs])

# Link library.
cmd = [linker] + sh_flags + ["-o"] + [libname] + ["-L%s" % cutest_libdir_double, "-lcutest_double"]
exit_code = subprocess.call(cmd)

#subprocess.call(['gfortran','-c','-fPIC','ELFUN.f','EXTER.f','GROUP.f','RANGE.f'])
#print(linker)
#subprocess.call([linker,sh_flags,"-o",libname+"."+ soname,"ELFUN.o","EXTER.o","GROUP.o","RANGE.o","-L"+ CUTEST_DIR+os.path.sep+"objects"+os.path.sep+CUTEST_ARCH+os.path.sep+"double"])
subprocess.call(['rm','ELFUN.f','EXTER.f','GROUP.f','RANGE.f','ELFUN.o','EXTER.o','GROUP.o','RANGE.o'])
if os.path.isfile("OUTSDIF.d") == False:
	raise AssertionError("File OUTSDIF.d not exist")
dir = os.getcwd()
os.chdir(dir[:-7])
subprocess.call(['python','setup.py','install'])
 	
