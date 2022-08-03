#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct
import argparse
import numpy as np

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


def parse_args ():

  description = 'Weights matrix converter from Python to C++'

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--filename', dest='filename', required=True, type=str, action='store', help='Weights file produced by Python simulations')
  parser.add_argument('--outdir', dest='outdir', required=False, type=str, action='store', help='Output directory', default='')

  args = parser.parse_args()

  return args

def main ():

  args = parse_args()

  name, ext = args.filename.split('.')[-2:]
  name = os.path.basename(name)

  if not args.outdir:
    args.outdir = os.path.dirname(os.path.abspath(__file__))

  outfile = os.path.join(args.outdir, '{}.bin'.format(name))

  with open(args.filename, 'rb') as fp:
    weights = np.fromfile(fp, dtype=np.float, count=-1)

  weights = weights.astype(np.float32)

  dims = weights.shape
  size = np.prod(dims)

  with open(outfile, 'wb') as fp:
    fp.write(struct.pack('ll', *dims))
    fp.write(struct.pack('f'*size, weights.ravel()))

if __name__ == '__main__':

  main()
