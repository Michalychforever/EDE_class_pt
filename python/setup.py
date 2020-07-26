from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as nm
import os
import subprocess as sbp
import os.path as osp

# Recover the gcc compiler
GCCPATH_STRING = sbp.Popen(
    ['gcc', '-print-libgcc-file-name'],
    stdout=sbp.PIPE).communicate()[0]
GCCPATH = osp.normpath(osp.dirname(GCCPATH_STRING)).decode()

liblist = ["class","openblas"]
MVEC_STRING = sbp.Popen(
    ['gcc', '-lmvec'],
    stderr=sbp.PIPE).communicate()[1]
if b"mvec" not in MVEC_STRING:
    liblist += ["mvec","m"]

# define absolute paths
root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
include_folder = os.path.join(root_folder, "include")
classy_folder = os.path.join(root_folder, "python")

# Recover the CLASS version
with open(os.path.join(include_folder, 'common.h'), 'r') as v_file:
    for line in v_file:
        if line.find("_VERSION_") != -1:
            # get rid of the " and the v
            VERSION = line.split()[-1][2:-1]
            break
# /usr/local/opt/openblas/lib/libopenblas.a
setup(
    name='classy',
    version=VERSION,
    description='Python interface to the Cosmological Boltzmann code CLASS',
    url='http://www.class-code.net',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("classy", [os.path.join(classy_folder, "classy.pyx")],
                           include_dirs=[nm.get_include(), include_folder],
                           libraries=liblist,
                           # library_dirs=[root_folder, GCCPATH, '/Users/michalychforever/Dropbox/Docs/science/OpenBLAS-0.2.20'],
                           # extra_link_args=['/Users/michalychforever/Dropbox/Docs/science/OpenBLAS-0.2.20/libopenblas.a','-lgomp'],
                           library_dirs=[root_folder, GCCPATH, '/usr/local/opt/openblas/lib'],
                           extra_link_args=['/usr/local/opt/openblas/lib/libopenblas.a','-lgomp'],
                           )],
    #data_files=[('bbn', ['../bbn/sBBN.dat'])]
)
