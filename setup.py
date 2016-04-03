from distutils.core import setup

setup(
    name = 'CUTEst.py',
    version='0.0.1',
    description='Python interface for CUTEst library',
    author=['Kenjy Demeester','Farooq Sanni'],
    package_dir={"cutest": "cutest"},
    package_data={'cutest': ['model/*.py','model/src/*.cfg','model/src/*.pyx','model/src/*.pxd','optimize/*py','tools/*.py']},
    packages=["cutest"],
    )
