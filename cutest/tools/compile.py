import os, sys, cutest, tempfile, subprocess, ConfigParser, numpy as np
from sys import platform
from distutils import sysconfig

CUTEST_ARCH = os.getenv('MYARCH')
CUTEST_DIR = os.getenv('CUTEST')
#CUTEST_LIB = os.getenv('CUTESTLIB')

if CUTEST_DIR is None or CUTEST_ARCH is None:
    raise(KeyError, "Please check that CUTEST and MYARCH are set")
cutest_libdir_double = os.path.join(CUTEST_DIR, "objects", CUTEST_ARCH, "double")

fcompiler = "gfortran"
ccompiler = "gcc"
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

def compile_SIF(problem):
    """Decode SIF problem and compile shared library."""

    cutest_config = ConfigParser.SafeConfigParser()
    
    #Find automatic the folder of site.cfg
    data_dir = os.path.join(os.path.dirname(cutest.__file__),'model','src')
    
    cutest_config.read(os.path.join(data_dir, 'site.cfg'))
    
    default_library_dir = cutest_config.get('DEFAULT', 'library_dirs').split(os.pathsep)
    default_include_dir = cutest_config.get('DEFAULT', 'include_dirs').split(os.pathsep)
    
    source1a = [os.path.join(data_dir, "ccutest.pyx")]
    source2a = [os.path.join(data_dir, "ccutest.pxd")]
    library_dirs = default_library_dir, 
    libraries = ['cutest']

    cur_path = os.getcwd()

    # Decode and compile problem in temprary directory.
    directory = tempfile.mkdtemp()
    os.chdir(directory) 

    # copy and paste .pxd and .pyx on the new directory 
    source1b = [os.path.join(directory, problem+".pyx")]
    source2b = [os.path.join(directory, problem+".pxd")]
    subprocess.call(['cp', source1a[0], source1b[0] ])
    subprocess.call(['cp', source2a[0], source2b[0] ])

    # Cythonize the .pyx to get the .c
    subprocess.call(['cython', '-I', library_dirs[0][0], problem+".pyx"])
    
    # Problem decode
    subprocess.call(['sifdecoder', problem])
    libname = "lib%s.%s" % (problem, soname)
    #Check if decode problem is succed
    if os.path.isfile("OUTSDIF.d") == False:
        sys.exit()

    srcs = ["ELFUN", "RANGE", "GROUP", "EXTER"]
    dat = ["OUTSDIF.d", "AUTOMAT.d"]

    # Create problem .o from .c
    subprocess.call([ccompiler,"-g","-O3","-fPIC","-I"+library_dirs[0][0], "-I"+np.get_include(),"-I"+sysconfig.get_python_inc(),"-c", problem+".c", "-o", problem+".o"])

    # Compile source files.
    exit_code = subprocess.call([fcompiler, "-c"] +  [src + ".f" for src in srcs])
    # Link library.
    cmd = [linker] + sh_flags + ["-o"] + [libname] + ["-L%s" % cutest_libdir_double, "-lcutest_double"]+ [src + ".o" for src in srcs]
    link_code = subprocess.call(cmd)

    # Link all problem library to create the .so
    if platform == "linux" or platform == "linux2": 
        cmd = [ccompiler] + sh_flags + [problem+".o"] + [src + ".o" for src in srcs] + ["-L"+lib for lib in library_dirs[0]] + ["-lcutest"] + ["-lgfortran"]+ ["-o"] + [problem +".so"]
    elif platform == "darwin":
        cmd = [ccompiler] + sh_flags + [problem+".o"] + [src + ".o" for src in srcs] + ["-L"+library_dirs[0][0]] + ["-lcutest"]+ ["-o"] + [problem +".so"]
    subprocess.call(cmd)

    # Clean the reposite
    subprocess.call(['rm','ELFUN.f','EXTER.f','GROUP.f','RANGE.f','ELFUN.o','EXTER.o','GROUP.o','RANGE.o'])
    
    os.chdir(cur_path)
    return directory


