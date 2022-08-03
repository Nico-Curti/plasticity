#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit
import itertools
import multiprocessing
from time import time as now

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

NUM_REPEATS = 3
NUMBER = 10


def pure_python_version (model, Nsample, Nfeature):

  SETUP_CODE = '''
from plasticity.model import {model}
from plasticity.model.optimizer import SGD
import numpy as np

optimizer = SGD(lr=2e-2)

data = np.random.uniform(low=-1., high=1., size=({Nsample}, {Nfeature}))
if "{model}" == 'BCM':
  model = {model}(outputs=100, num_epochs=10, batch_size=10, activation='Logistic',
                  optimizer=optimizer, mu=0., sigma=1., interaction_strength=0.,
                  precision=1e-30, seed=42, verbose=False)
elif "{model}" == 'Hopfield':
  model = {model}(outputs=100, num_epochs=10, batch_size=10, delta=0.4,
                  optimizer=optimizer, mu=0., sigma=1.,
                  p=2., k=2, precision=1e-30, seed=42, verbose=False)
else:
  raise ValueError
  '''.format(**{'model' : model, 'Nsample' : Nsample, 'Nfeature' : Nfeature})

  TEST_CODE = '''
model.fit(data)
  '''.format()

  times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=NUM_REPEATS, number=NUMBER)

  return times

def cython_version (model, Nsample, Nfeature):

  SETUP_CODE = '''
from plasticity.cython import {model}
from plasticity.cython.optimizer import SGD
import numpy as np

optimizer = SGD(learning_rate=2e-2)

data = np.random.uniform(low=-1., high=1., size=({Nsample}, {Nfeature}))
if "{model}" == 'BCM':
  model = {model}(outputs=100, num_epochs=10, batch_size=10, activation='Logistic',
                  optimizer=optimizer, mu=0., sigma=1., interaction_strength=0.,
                  seed=42)
elif "{model}" == 'Hopfield':
  model = {model}(outputs=100, num_epochs=10, batch_size=10, delta=0.4,
                  optimizer=optimizer, mu=0., sigma=1.,
                  p=2., k=2, seed=42)
else:
  raise ValueError
  '''.format(**{'model' : model, 'Nsample' : Nsample, 'Nfeature' : Nfeature})

  TEST_CODE = '''
model.fit(data)
  '''.format()

  times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=NUM_REPEATS, number=NUMBER)

  return times



if __name__ == '__main__':

  import numpy as np

  Nsamples = [10, 20, 40, 60, 80, 100]
  Nfeatures = [100, 250, 500, 750, 1000]
  model = 'Hopfield'

  nth = multiprocessing.cpu_count()

  parameters = itertools.product(Nsamples, Nfeatures)

  with open ('plasticity_{}_timing.dat'.format(model), 'w') as fp:

    row = 'num_repeats,number,samples,features,nth,py_mean,py_max,py_min,py_std,cy_mean,cy_max,cy_min,cy_std'
    fp.write('{}\n'.format(row))

    for (Nsample, Nfeature) in parameters:

      print('Evaluating (sample={}, feature={}) ...'.format(Nsample, Nfeature),
           end='', flush=True)

      py_tic = now()

      times = pure_python_version(model, Nsample, Nfeature)

      py_mean = np.mean(times)
      py_max  = np.max(times)
      py_min  = np.min(times)
      py_std  = np.std(times)

      py_toc = now()

      cy_tic = now()

      times = cython_version(model, Nsample, Nfeature)

      cy_mean = np.mean(times)
      cy_max  = np.max(times)
      cy_min  = np.min(times)
      cy_std  = np.std(times)

      cy_toc = now()

      save = '{NUM_REPEATS},{NUMBER},{Nsample},{Nfeature},{NTH},\
{PY_MEAN:.3f},{PY_MAX:.3f},{PY_MIN:.3f},{PY_STD:.3f},\
{CY_MEAN:.3f},{CY_MAX:.3f},{CY_MIN:.3f},{CY_STD:.3f}'.format(
              **{'NUM_REPEATS':NUM_REPEATS, 'NUMBER':NUMBER,
                 'Nsample':Nsample, 'Nfeature':Nfeature, 'NTH':nth,
                 'PY_MEAN':py_mean, 'PY_MAX':py_max, 'PY_MIN':py_min, 'PY_STD':py_std,
                 'CY_MEAN':cy_mean, 'CY_MAX':cy_max, 'CY_MIN':cy_min, 'CY_STD':cy_std
                 })

      fp.write('{}\n'.format(save))

      print('  took {:.3f} seconds'.format(cy_toc - py_tic), flush=True)
