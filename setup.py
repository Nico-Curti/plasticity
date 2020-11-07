#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
import platform
import subprocess

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

def get_ext_filename_without_platform_suffix (filename):
  name, ext = os.path.splitext(filename)
  ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

  if ext_suffix == ext:
    return filename

  ext_suffix = ext_suffix.replace(ext, '')
  idx = name.find(ext_suffix)

  if idx == -1:
    return filename
  else:
    return name[:idx] + ext


class custom_build_ext (build_ext):
  '''
  Custom build type
  '''

  def get_ext_filename (self, ext_name):

    if platform.system() == 'Windows':
      # The default EXT_SUFFIX of windows includes the PEP 3149 tags of compiled modules
      # In this case I rewrite a custom version of the original distutils.command.build_ext.get_ext_filename function
      ext_path = ext_name.split('.')
      ext_suffix = '.pyd'
      filename = os.path.join(*ext_path) + ext_suffix
    else:
      filename = super().get_ext_filename(ext_name)

    return get_ext_filename_without_platform_suffix(filename)

  def build_extensions (self):

    customize_compiler(self.compiler)

    try:
      self.compiler.compiler_so.remove('-Wstrict-prototypes')

    except (AttributeError, ValueError):
      pass

    build_ext.build_extensions(self)


class sdist (_sdist):

  def run (self):

    self.run_command('build_ext')
    _sdist.run(self)


def get_eigen_link_flags ():
  '''
  Get the links and flags variables for Eigen links
  '''

  # TODO: this solution could work only for UNIX OS
  # A possible workaround is given by scikit-build!
  # I will move the setup to scikit-build ASAP

  eigen_lib_flags = subprocess.run('echo `pkg-config --cflags eigen3`',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True
                                    )
  if eigen_lib_flags.stderr:

    # Check if we're running on Read the Docs' servers
    read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

    if read_the_docs_build:
      eigen_lib_flags = '-I/include/eigen3/'
    else:
      raise OSError('Package eigen3 was not found in the pkg-config search path.')
  else:
    eigen_lib_flags = eigen_lib_flags.stdout[:-1]

  return eigen_lib_flags


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
EMAIL = 'nico.curti2@unibo.it, simone.gasperini2@studio.unibo.it, mattia.ceccarelli5@unibo.it'
AUTHOR = 'Nico Curti, Simone Gasperini, Mattia Ceccarelli'
REQUIRES_PYTHON = '>=3.5'
VERSION = None
KEYWORDS = 'neural-networks deep-neural-networks deep-learning image-classification'

CPP_COMPILER = platform.python_compiler()
README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'plasticity', '__version__.py')

ENABLE_OMP = False

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

# link to OpenCV libraries
eigen_flags = get_eigen_link_flags()

define_args = [ '-DMAJOR={}'.format(Version[0]),
                '-DMINOR={}'.format(Version[1]),
                '-DREVISION={}'.format(Version[2])
                ]


if 'GCC' in CPP_COMPILER or 'Clang' in CPP_COMPILER:

  cpp_compiler_args = ['-std=c++1z', '-std=gnu++1z', '-g0']

  compile_args = ['-Wno-unused-function', # disable unused-function warnings
                  '-Wno-narrowing', # disable narrowing conversion warnings
                   # enable common warnings flags
                  '-Wall',
                  '-Wextra',
                  '-Wno-unused-result',
                  '-Wno-unknown-pragmas',
                  '-Wfatal-errors',
                  '-Wpedantic',
                  '-march=native',
                  '-Ofast'
                  ]

  try:

    compiler, compiler_version = CPP_COMPILER.split()

  except ValueError:

    compiler, compiler_version = (CPP_COMPILER, '0')

  if ENABLE_OMP and compiler == 'GCC':
    linker_args = [eigen_flags, '-fopenmp']

  else:
    linker_args = [eigen_flags, ]

  if 'Clang' in CPP_COMPILER and 'clang' in os.environ['CXX']:
    cpp_compiler_args += ['-stdlib=libc++']

elif 'MSC' in CPP_COMPILER:
  cpp_compiler_args = ['/std:c++latest']
  compile_args = []

  if ENABLE_OMP:
    linker_args = [eigen_flags, '/openmp']
  else:
    linker_args = [eigen_flags, ]

else:
  raise ValueError('Unknown c++ compiler arg')

whole_compiler_args = sum([cpp_compiler_args, compile_args, define_args, linker_args], [])

cmdclass = {'build_ext': custom_build_ext,
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
                                    Extension(name='.'.join(['plasticity', 'lib', 'bcm']),
                                              sources=['./plasticity/source/bcm.pyx',
                                                       './src/activations.cpp',
                                                       './src/fmath.cpp',
                                                       './src/bcm.cpp',
                                                       './src/base.cpp',
                                                       './src/utils.cpp',
                                              ],
                                              include_dirs=['./plasticity/lib/',
                                                            './hpp/',
                                                            './include/'
                                              ],
                                              libraries=[],
                                              library_dirs=[
                                                            os.path.join(here, 'lib'),
                                                            os.path.join('usr', 'lib'),
                                                            os.path.join('usr', 'local', 'lib'),
                                              ],  # path to .a or .so file(s)
                                              extra_compile_args = whole_compiler_args,
                                              extra_link_args = linker_args,
                                              language='c++'
                                              ),

                                    Extension(name='.'.join(['plasticity', 'lib', 'hopfield']),
                                              sources=['./plasticity/source/hopfield.pyx',
                                                       './src/activations.cpp',
                                                       './src/base.cpp',
                                                       './src/fmath.cpp',
                                                       './src/hopfield.cpp',
                                                       './src/utils.cpp'
                                              ],
                                              include_dirs=['./plasticity/lib/',
                                                            './hpp/',
                                                            './include/'
                                              ],
                                              libraries=[],
                                              library_dirs=[
                                                            os.path.join(here, 'lib'),
                                                            os.path.join('usr', 'lib'),
                                                            os.path.join('usr', 'local', 'lib'),
                                              ],  # path to .a or .so file(s)
                                              extra_compile_args = whole_compiler_args,
                                              extra_link_args = linker_args,
                                              language='c++'
                                              ),
  ],
)
