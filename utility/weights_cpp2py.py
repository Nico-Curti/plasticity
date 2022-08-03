#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct
import argparse
import numpy as np

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


def parse_args ():

  description = 'Weights matrix converter from C++ to Python'

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--filename', dest='filename', required=True, type=str, action='store', help='Weights file produced by C++ simulations')
  parser.add_argument('--outdir', dest='outdir', required=False, type=str, action='store', help='Output directory', default='')

  args = parser.parse_args()

  return args

def main ():

  args = parse_args()

  name, ext = args.filename.split('.')[-2:]
  name = os.path.basename(name)

  if not args.outdir:
    args.outdir = os.path.dirname(os.path.abspath(__file__))

  outfile = os.path.join(args.outdir, '{}.pkl'.format(name))

  with open(args.filename, 'rb') as fp:
    dims = struct.unpack('ll', fp.read(16))
    size = np.prod(dims)
    weights = struct.unpack('f'*size, fp.read(4*size))
    weights = np.asarray(weights).reshape(dims)

  with open(outfile, 'wb') as fp:
    weights.tofile(fp, sep='')

if __name__ == '__main__':

  main()
