import os
import sys
import platform
import subprocess
try:
    from  cutest.ccutest import  loadCutestProb
except ImportError:
    print("Dynamic library problem unfound, please compiled with creatlib.py")
    sys.exit(0)

	
if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    loadCutestProb(name)
