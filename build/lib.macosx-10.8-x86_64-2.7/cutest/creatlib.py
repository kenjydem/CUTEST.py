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

def compile_sif_problem(problem):
    """Decode SIF problem and compile shared library."""
    import tempfile

    cur_path = os.getcwd()

    # Decode and compile problem in temprary directory.
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    if problem[-4:] == ".SIF":
        problem = problem[:-4]
    subprocess.call(['sifdecoder', problem])
    libname = "lib%s.%s" % (problem, soname)

    srcs = ["ELFUN", "RANGE", "GROUP", "EXTER"]
    dat = ["OUTSDIF.d", "AUTOMAT.d"]

    # Compile source files.
    exit_code = subprocess.call([fcompiler, "-c"] +  [src + ".f" for src in srcs])

    # Link library.
    cmd = [linker] + sh_flags + ["-o"] + [libname] + ["-L%s" % cutest_libdir_double, "-lcutest_double"] + [src + ".o" for src in srcs]
    link_code = subprocess.call(cmd)

    subprocess.call(['rm','ELFUN.f','EXTER.f','GROUP.f','RANGE.f','ELFUN.o','EXTER.o','GROUP.o','RANGE.o'])

    if os.path.isfile("OUTSDIF.d") == False:
        raise AssertionError("File OUTSDIF.d not exist")

    os.chdir(cur_path)
    return (tmpdir, libname)
#dir = os.getcwd()
#os.chdir(dir[:-7])
#subprocess.call(['python','setup.py','install'])
 	
