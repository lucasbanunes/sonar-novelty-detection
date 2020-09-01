"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import gc
import multiprocessing
import argparse

import numpy as np

from config import setup
#Gets enviroment variables and appends needed custom packages
setup()
from neural_networks.evaluation import eval_all_novelties

LINES = '--------------------------------------------------------------------\n'

parser = argparse.ArgumentParser(description='Evaluates the results of a neural network training',
                                    prog='Neual Network Kfold evaluation')
parser.add_argument('dir', help='Directory generated from nn_kfold.py with the results and training history')

OUTPUT_DIR = parser.parse_args().dir

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    if not os.path.isdir(current_dir):
        continue
    
    splitted = lofar_params_folder.split('_')
    fft_pts = splitted[2]
    decimation = splitted[5]
    pca = splitted[-3]
    bins = splitted[-1]

    model_neurons_folders = np.sort(os.listdir(current_dir))
    add_dir = lambda x: os.path.join(current_dir, x)
    func_args = filter(os.path.isdir, map(add_dir, model_neurons_folders))

    print(LINES + f'fft_pts: {fft_pts}\n decimation: {decimation}\n' + LINES)
    pool = multiprocessing.Pool()
    pool.map(eval_all_novelties, func_args)
    pool.close()
    pool.join()
    del pool

    current_dir, _ = os.path.split(current_dir)