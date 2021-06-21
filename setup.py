#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
import sys
import pathlib
import platform
import numpy as np

try:
  from setuptools import setup
  from setuptools import Extension
  from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import Extension
  from distutils.core import find_packages

from distutils import sysconfig
from Cython.Distutils import build_ext
from distutils.sysconfig import customize_compiler
from distutils.command.sdist import sdist as _sdist


class CMakeExtension (Extension):

  # Reference: https://stackoverflow.com/a/48015772

  def __init__(self, name):
    # don't invoke the original build_ext for this special extension
    super().__init__(name, sources=[])


class cmake_build_ext (build_ext):

  # Reference: https://stackoverflow.com/a/48015772

  def run (self):

    for ext in self.extensions:
      self.build_cmake(ext)

    super().run()

  def build_cmake (self, ext):

    cwd = pathlib.Path().absolute()

    # these dirs will be created in build_py, so if you don't have
    # any python sources to bundle, the dirs will be missing
    build_temp = pathlib.Path(self.build_temp)
    build_temp.mkdir(parents=True, exist_ok=True)
    extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
    extdir.mkdir(parents=True, exist_ok=True)

    # example of cmake args
    config = 'Debug' if self.debug else 'Release'
    cmake_args = [
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:FILEPATH=' + str(extdir.parent.absolute()) + '/lib',
        '-DCMAKE_BUILD_TYPE:STRING=' + config,
        '-DPYWRAP:BOOL=ON',
        '-DBUILD_DOCS:BOOL={}'.format('ON' if os.environ.get('READTHEDOCS', None) == 'True' else 'OFF'),
        '-DBUILD_TEST:BOOL=OFF',
        '-DVERBOSE:BOOL=ON',
        '-DVIEW:BOOL=OFF',
        '-DPython3_EXECUTABLE:FILEPATH={}'.format(sys.executable),
        '-DPython3_INCLUDE_DIR:FILEPATH={}'.format(sysconfig.get_python_inc()),
        '-DPython3_NumPy_INCLUDE_DIR:FILEPATH={}'.format(np.get_include()),
        '-DOMP:BOOL={}'.format('ON' if ENABLE_OMP else 'OFF'),
    ]

    if platform.system() == 'Windows':
      vcpkg_root = os.environ.get('VCPKG_ROOT').replace('\\', '/')

      if not vcpkg_root:
        raise ValueError('VCPKG not found. '
                         'Please set the environment variable to the path in which vcpkg can be found. '
                         '(E.g $env:VCPKG_ROOT=C:/Users/Myuser/vcpkg/')

      vcpk_triplet = os.environ.get('VCPKG_DEFAULT_TRIPLET')
      vcpk_triplet = vcpk_triplet if vcpk_triplet else 'x64-windows'

      cmake_args.extend(['-DCMAKE_TOOLCHAIN_FILE={}/scripts/buildsystems/vcpkg.cmake'.format(vcpkg_root),
                         '-DVCPKG_TARGET_TRIPLET={}'.format(vcpk_triplet)
                         ])

    # example of build args
    build_args = [
        '--target', 'install',
        '--config', config,
        '--parallel', '4',
    ]

    os.chdir(str(build_temp))
    self.spawn(['cmake', str(cwd)] + cmake_args)
    if not self.dry_run:
      self.spawn(['cmake', '--build', '.'] + build_args)
    # Troubleshooting: if fail on line above then delete all possible
    # temporary CMake files including "CMakeCache.txt" in top level dir.
    os.chdir(str(cwd))



class sdist (_sdist):

  def run (self):

    self.run_command('build_ext')
    _sdist.run(self)



def get_requires (requirements_filename):
  '''
  What packages are required for this module to be executed?

  Parameters
  ----------
    requirements_filename : str
      filename of requirements (e.g requirements.txt)

  Returns
  -------
    requirements : list
      list of required packages
  '''
  with open(requirements_filename, 'r') as fp:
    requirements = fp.read()

  return list(filter(lambda x: x != '', requirements.split()))


def read_description (readme_filename):
  '''
  Description package from filename

  Parameters
  ----------
    readme_filename : str
      filename with readme information (e.g README.md)

  Returns
  -------
    description : str
      str with description
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

    return description

  except IOError:
    return ''

def read_version (CMakeLists):
  '''
  Read version from variables set in CMake file

  Parameters
  ----------
    CMakeLists : string
      Main CMakefile filename or path

  Returns
  -------
    version : tuple
      Version as (major, minor, revision) of strings
  '''
  major = re.compile(r'set\s+\(PLASTICITY_MAJOR\s+(\d+)\)')
  minor = re.compile(r'set\s+\(PLASTICITY_MINOR\s+(\d+)\)')
  revision = re.compile(r'set\s+\(PLASTICITY_REVISION\s+(\d+)\)')

  with open(CMakeLists, 'r') as fp:
    cmake = fp.read()

  major_v = major.findall(cmake)[0]
  minor_v = minor.findall(cmake)[0]
  revision_v = revision.findall(cmake)[0]

  version = map(int, (major_v, minor_v, revision_v))

  return tuple(version)


def dump_version_file (here, version_filename):
  '''
  Dump the __version__.py file as python script

  Parameters
  ----------
    here : string
      Local path where the CMakeLists.txt file is stored

    version_filename: string
      Filename or path where to save the __version__.py filename
  '''

  VERSION = read_version(os.path.join(here, './CMakeLists.txt'))

  script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__  = ['Nico Curti', 'SimoneGasperini', 'Mattia Ceccarelli']
__email__ = ['nico.curti2@unibo.it', 'simone.gasperini2@studio.unibo.it', 'mattia.ceccarelli5@unibo.it']

__version__ = '{}.{}.{}'
'''.format(*VERSION)

  with open(version_filename, 'w') as fp:
    fp.write(script)


here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = 'plasticity'
DESCRIPTION = 'Unsupervised Neural Networks with biological-inspired learning rules'
URL = 'https://github.com/Nico-Curti/plasticity'
EMAIL = 'nico.curti2@unibo.it, lorenzo.squadrani@studio.unibo.it, simone.gasperini2@studio.unibo.it, mattia.ceccarelli5@unibo.it'
AUTHOR = 'Nico Curti, Lorenzo Squadrani, Simone Gasperini, Mattia Ceccarelli'
REQUIRES_PYTHON = '>=3.5'
VERSION = None
KEYWORDS = 'neural-networks deep-neural-networks deep-learning image-classification'

CPP_COMPILER = platform.python_compiler()
README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'plasticity', '__version__.py')

ENABLE_OMP = False

if '--omp' in sys.argv:
  ENABLE_OMP = True
  sys.argv.remove('--omp')

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except IOError:
  LONG_DESCRIPTION = DESCRIPTION

dump_version_file(here, VERSION_FILENAME)

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
  with open(VERSION_FILENAME) as fp:
    exec(fp.read(), about)

else:
  about['__version__'] = VERSION

# parse version variables and add them to command line as definitions
Version = about['__version__'].split('.')

cmdclass = {'build_ext': cmake_build_ext,
            'sdist': sdist}


setup(
  name                          = NAME,
  version                       = about['__version__'],
  description                   = DESCRIPTION,
  long_description              = LONG_DESCRIPTION,
  long_description_content_type = 'text/markdown',
  author                        = AUTHOR,
  author_email                  = EMAIL,
  maintainer                    = AUTHOR,
  maintainer_email              = EMAIL,
  python_requires               = REQUIRES_PYTHON,
  install_requires              = get_requires(REQUIREMENTS_FILENAME),
  url                           = URL,
  download_url                  = URL,
  keywords                      = KEYWORDS,
  setup_requires                = [# Setuptools 18.0 properly handles Cython extensions.
                                   'setuptools>=18.0',
                                   'cython'],
  packages                      = find_packages(include=['plasticity', 'plasticity.*'], exclude=('test', 'testing')),
  include_package_data          = True, # no absolute paths are allowed
  platforms                     = 'any',
  classifiers                   = [
                                   #'License :: OSI Approved :: GPL License',
                                   'Natural Language :: English',
                                   'Operating System :: MacOS :: MacOS X',
                                   'Operating System :: POSIX',
                                   'Operating System :: POSIX :: Linux',
                                   'Operating System :: Microsoft :: Windows',
                                   'Programming Language :: Python',
                                   'Programming Language :: Python :: 3',
                                   'Programming Language :: Python :: 3.5',
                                   'Programming Language :: Python :: 3.6',
                                   'Programming Language :: Python :: 3.7',
                                   'Programming Language :: Python :: 3.8',
                                   'Programming Language :: Python :: Implementation :: CPython',
                                   'Programming Language :: Python :: Implementation :: PyPy'
                                  ],
  license                       = 'MIT',
  cmdclass                      = cmdclass,
  ext_modules                   = [
                                    CMakeExtension(name=NAME)
  ],
)
