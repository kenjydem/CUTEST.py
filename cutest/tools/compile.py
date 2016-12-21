import os, sys, cutest, tempfile, subprocess, ConfigParser, numpy as np
from sys import platform
from distutils import sysconfig

CUTEST_ARCH = os.getenv('MYARCH')
CUTEST_DIR = os.getenv('CUTEST')

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

def compile_SIF(problem_name, sifParams):
    """Decode SIF problem and compile shared library."""

    cutest_config = ConfigParser.SafeConfigParser()
    
    #Find automatic the folder of site.cfg
    data_dir = os.path.join(os.path.dirname(cutest.__file__),'model','src')
    config_dir = os.path.join(os.path.dirname(cutest.__file__),'tools')

    cutest_config.read(os.path.join(config_dir, 'site.template.cfg'))
    
    default_library_dir = cutest_config.get('DEFAULT', 'library_dirs').split(os.pathsep)
    default_include_dir = cutest_config.get('DEFAULT', 'include_dirs').split(os.pathsep)
    
    source1a = [os.path.join(data_dir, "cutest.pyx")]
    source2a = [os.path.join(data_dir, "cutest.pxd")]
    library_dirs = default_library_dir, 
    libraries = ['cutest']

    cur_path = os.getcwd()

    # Decode and compile problem in temprary directory.
    directory = tempfile.mkdtemp()
    os.chdir(directory) 

    # Check problem name
    if problem_name is "10FOLDTR":
        problem_name_cython = "FOLDTR"
    elif problem_name is "3PK":
        problem_name_cython = "PK"
    elif "-" in problem_name:
        ind = problem_name.find("-")
        problem_name_cython = problem_name[:ind]+problem_name[ind+1:]
    else:
        problem_name_cython = problem_name


    # copy and paste .pxd and .pyx on the new directory 
    source1b = [os.path.join(directory, problem_name_cython + ".pyx")]
    source2b = [os.path.join(directory, problem_name_cython + ".pxd")]
    subprocess.call(['cp', source1a[0], source1b[0] ])
    subprocess.call(['cp', source2a[0], source2b[0] ])

    # Cythonize the .pyx to get the .c
    subprocess.call(['cython', '-I', library_dirs[0][0], problem_name_cython+".pyx"])
    
    # Problem decode
    if sifParams is None:
        subprocess.call(['sifdecoder', problem_name])
    else:
        cmd = ['sifdecoder']+ [param for param in sifParams]+ [problem_name]
        subprocess.call(cmd)
     
    libname = "lib%s.%s" % (problem_name, soname)
    #Check if decode problem is succed
    if os.path.isfile("OUTSDIF.d") == False:
        sys.exit()

    srcs = ["ELFUN", "RANGE", "GROUP", "EXTER"]
    dat = ["OUTSDIF.d", "AUTOMAT.d"]

    # Create problem .o from .c
    subprocess.call([ccompiler,"-w","-g","-O3","-fPIC","-I"+library_dirs[0][0], "-I"+np.get_include(),"-I"+sysconfig.get_python_inc(),"-c", problem_name_cython+".c", "-o", problem_name_cython+".o"])

    # Compile source files.
    exit_code = subprocess.call([fcompiler, "-c", "-fPIC"] +  [src + ".f" for src in srcs])
    # Link library.
    cmd = [linker] + sh_flags + ["-o"] + [libname] + ["-L%s" % cutest_libdir_double, "-lcutest_double"]+ [src + ".o" for src in srcs]
    link_code = subprocess.call(cmd)

    # Link all problem library to create the .so
    if platform == "linux" or platform == "linux2": 
        cmd = [ccompiler] + sh_flags  + [problem_name_cython+".o"] + [src + ".o" for src in srcs] + ["-L"+lib for lib in library_dirs[0]] + ["-lcutest"] + ["-lgfortran"]+ ["-o"] + [problem_name_cython +".so"]
    elif platform == "darwin":
        cmd = [ccompiler] + sh_flags + [problem_name_cython+".o"] + [src + ".o" for src in srcs] + ["-L"+library_dirs[0][0]] + ["-lcutest"]+ ["-o"] + [problem_name_cython +".so"]
    subprocess.call(cmd)

    # Clean the reposite
    subprocess.call(['rm','ELFUN.f','EXTER.f','GROUP.f','RANGE.f','ELFUN.o','EXTER.o','GROUP.o','RANGE.o'])
    
    os.chdir(cur_path)

    return directory, problem_name_cython


