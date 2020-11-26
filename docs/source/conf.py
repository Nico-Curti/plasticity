# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

sys.path.insert(0, os.path.abspath('../../plasticity/'))

def read_version (CMakeLists):
  '''
  Read version from variables set in CMake file

  Parameters
  ----------
  CMakeLists : CMake file path

  Returns
  -------
  version : tuple
    Version as (major, minor, revision)
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

try:

  LOCAL = os.path.dirname(__file__)

except NameError:

  LOCAL = ''

VERSION = read_version(os.path.join(LOCAL, '..', '..', 'CMakeLists.txt'))
__version__ = '.'.join(map(str, VERSION))

# -- Project information -----------------------------------------------------

project = 'plasticity - Unsupervised Neural Networks with biological-inspired learning rules'
copyright = '2020, Nico Curti, Simone Gasperini, Mattia Ceccarelli'
author = 'Nico Curti, Simone Gasperini, Mattia Ceccarelli'

# The full version, including alpha/beta/rc tags
release = __version__

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'breathe',
              'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_kernel_name = 'python3'
nbsphinx_output_prompt = 'Out[%s]:'


breathe_projects = {
  'Activations' : './doxydoc/',
  'BasePlasticity' : './doxydoc/',
  'BCM' : './doxydoc/',
  'Hopfield' : './doxydoc/',
  'data_dispatcher' : './doxydoc/',
  }
