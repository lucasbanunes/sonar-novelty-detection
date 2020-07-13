"""Reads the frame generated from best_selection.py and plots the results with error bars."""
import os
import gc
import sys
import pdb
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from setup import setup
from sklearn.metrics import recall_score, accuracy_score

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()

import my_models
from data_analysis.model_evaluation import novelty_analysis
import data_analysis.utils.utils as da_utils

#COLORS = (black, gray, brown) in hex code
COLORS = ('#000000', '#808080', '#A52A2A')
WIDTH = 0.1

best_models = pd.read_csv(os.path.join(OUTPUT_DIR, 'best_models.csv'), header=[0,1], index_col=[0,1])

fft_pts = [1024,2048,4096]
decimations = [0,3,5]
positions = ['First', 'Second', 'Third']
COLORS_PER_POSITION = {position: color for position, color in zip(positions, COLORS)}
rows = ['_'.join([position, color]) for position, color in zip(positions, COLORS)]

def _error_line(axis, x, y, error, width, color):
    x_min = x-width/2
    x_max = x+width/2
    axis.scatter(x, y, c=color)
    axis.hlines(y-error, x_min, x_max, colors=color)
    axis.hlines(y+error, x_min, x_max, colors=color)
    axis.vlines(x, y-error, y+error, colors=color)

def error_lines(axis, x, y, error, width, color):
    for i in range(len(x)):
        _error_line(axis, x[i], y[i],error[i], width, color)
    return axis

for fft in fft_pts:
    fig, axis = plt.subplots()
    cell_text = list()
    for position in positions:
        cell_text.append(list(best_models.loc[fft, (position, 'Model')].values))
        error_lines(axis, [0,1,2], best_models.loc[fft, (position, 'Average')].values,
                    best_models.loc[fft, (position, 'Error')].values, WIDTH, COLORS_PER_POSITION[position])
    axis.set_ylabel('NOC AUC')
    axis.set_title(f'LOFAR com {fft} FFT pts')
    axis.set_xticks([])
    row_labels = ['preto', 'cinza', 'marrom']
    axis.table(cellText=cell_text,
                      rowLabels=row_labels,
                      colLabels=decimations,
                      loc='bottom')
    fig.savefig(os.path.join(OUTPUT_DIR, f'plot_best_selection_{fft}_fft_pts.png'), dpi=200, format='png', bbox_inches="tight")