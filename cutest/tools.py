import os
import sys
import subprocess
from sys import platform

CUTEST_ARCH = os.getenv('MYARCH')
CUTEST_DIR = os.getenv('CUTEST')
CUTEST_LIB = os.getenv('CUTESTLIB')

if CUTEST_DIR is None or CUTEST_ARCH is None:
    raise(KeyError, "Please check that CUTEST and MYARCH are set")
cutest_libdir_double = os.path.join(CUTEST_DIR, "objects", CUTEST_ARCH, "double")

fcompiler = "gfortran"
if platform == "linux" or platform == "linux2":
    linker = "ld"
    sh_flags = ["-shared"]
    soname = "so"
elif platform == "darwin":
    linker = "gfortran"
    sh_flags = ["-dynamiclib", "-undefined", "dynamic_lookup"]
    soname = "dylib"
elif platform == "win32":
    raise(ValueError, "Windows platforms are currently not supported")

def compile(problem):
    """Decode SIF problem and compile shared library."""
    #import tempfile

    cur_path = os.getcwd()

    # Decode and compile problem in temprary directory.
    os.chdir(CUTEST_LIB) 

    directory = os.path.join(CUTEST_LIB,problem)

    current_path = os.environ['DYLD_LIBRARY_PATH'] 
    os.environ['DYLD_LIBRARY_PATH'] = directory + ':' + current_path

    if not os.path.exists(directory):
        os.makedirs(problem)
    os.chdir(directory)    
    subprocess.call(['sifdecoder', problem])
    libname = "lib%s.%s" % (problem, soname)

    srcs = ["ELFUN", "RANGE", "GROUP", "EXTER"]
    dat = ["OUTSDIF.d", "AUTOMAT.d"]

    # Compile source files.
    exit_code = subprocess.call([fcompiler, "-c"] +  [src + ".f" for src in srcs])
    # Link library.
    cmd = [linker] + sh_flags + ["-o"] + [libname] + ["-L%s" % cutest_libdir_double, "-lcutest_double"]+ [src + ".o" for src in srcs]
    link_code = subprocess.call(cmd)

    os.system("gcc-5 -bundle -undefined dynamic_lookup -O3 -fPIC ~/Desktop/cutest/build/temp.macosx-10.8-x86_64-2.7/cutest/ccutest.o -L/Users/kenjydemeester/Desktop/HS10 -L/usr/local/lib/ -L/usr/local/lib -L/usr/local/opt/openssl/lib -L/usr/local/opt/sqlite/lib -lHS10 -lcutest -o ccutest.so")

    subprocess.call(['rm','ELFUN.f','EXTER.f','GROUP.f','RANGE.f','ELFUN.o','EXTER.o','GROUP.o','RANGE.o'])

    if os.path.isfile("OUTSDIF.d") == False:
        raise AssertionError("File OUTSDIF.d not exist")

    os.chdir(cur_path)
    return directory


