#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pylab as plt
import seaborn as sns

import matplotlib.patches as mpatches

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  model = 'Hopfield'
  filename = 'plasticity_{}_timing.dat'.format(model)

  data = pd.read_csv(filename, sep=',')

  labels = [ mpatches.Patch(facecolor='steelblue', label='pure-Python', edgecolor='k', linewidth=2),
             mpatches.Patch(facecolor='orange', label='Cython OMP', edgecolor='k', linewidth=2)
           ]

  with sns.plotting_context('paper', font_scale=2):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle('{} model'.format(model))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    temp = data.query('features == 1000')

    sns.regplot(x='samples', y='py_mean', data=data, x_estimator=plt.mean, ax=ax1)
    sns.regplot(x='samples', y='cy_mean', data=data, x_estimator=plt.mean, ax=ax1)

    ax1.legend(handles=labels, loc='upper left')
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Time (sec)')
    ax1.set_xlim(9, 101)

    sns.despine(ax=ax1, offset=10, top=True, right=True, bottom=False, left=False)

    temp = data.query('samples == 100')

    sns.regplot(x='features', y='py_mean', data=data, x_estimator=plt.mean, ax=ax2)
    sns.regplot(x='features', y='cy_mean', data=data, x_estimator=plt.mean, ax=ax2)

    ax2.legend(handles=labels, loc='upper left')
    ax2.set_xlabel('Number of features')
    ax2.set_ylabel('Time (sec)')
    ax2.set_xlim(90, 1010)

    sns.despine(ax=ax2, offset=10, top=True, right=True, bottom=False, left=False)

    fig.savefig('{}_timing.png'.format(model), bbox_inches='tight', dpi=300)
