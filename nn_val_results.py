"""This script mounts a results_frame for validation data"""

import os
import gc
import multiprocessing
import argparse

import numpy as np
import tensorflow as tf

from config import setup
_, LOFAR_FOLDER, _ = setup()
from neural_networks import evaluation as nn_evaluation
from data_analysis.model_evaluation.novelty_analysis import create_threshold

LINES = '--------------------------------------------------------------------\n'

parser = argparse.ArgumentParser(description='Evaluates the results of a neural network training',
                                    prog='Neual Network Kfold evaluation')
parser.add_argument('dir', help='Directory generated from nn_kfold.py with the results and training history')

gpus = gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
OUTPUT_DIR = parser.parse_args().dir
PCA = None
OVERLAP = 0
WINDOW_SIZE = 10
STRIDE = 5
N_FOLDS = 5
NON_WINDOWED_MODELS = ['mlp', 'mlp_expert']
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])       #Name of the classes
CLASSES_VALUES = np.arange(len(CLASSES_NAMES))      #Value of each class to be used in the model
THRESHOLD = create_threshold((200,450), ((-1,0.5),(0.500000001,1)))

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

    for model_neurons_folder in model_neurons_folders:
        current_dir = os.path.join(current_dir, model_neurons_folder)
        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        method_novelty_folders = np.sort(os.listdir(current_dir))
        
        splitted = model_neurons_folder.split('_')
        splitted_name = list()
        for string in splitted:
            if string == 'intermediate':
                break
            splitted_name.append(string)
        model_name = '_'.join(splitted_name)
        num_neurons = splitted[splitted.index('neurons') + 1]

        if model_name[:3] == 'old':
            current_dir, _ = os.path.split(current_dir)
            continue
        elif model_name == 'neural_committee':
            in_model_splitted = splitted[splitted.index('neurons') + 3:]
            input_model = '_'.join(in_model_splitted[: in_model_splitted.index('neurons')])
            input_neurons = in_model_splitted[-1]
        else:
            current_dir, _ = os.path.split(current_dir)
            input_model = model_name
            continue

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds (folder with all the folds)

            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))

            novelty_class = method_novelty_folder[-1]
            nov_index = list(CLASSES_NAMES).index(novelty_class)
            fold_count = 1
            if input_model in NON_WINDOWED_MODELS:
                windowed = False
                datapath = os.path.join(LOFAR_FOLDER, 
                                f'lofar_dataset_fft_pts_{fft_pts}_overlap_{OVERLAP}_decimation_{decimation}_pca_{PCA}_novelty_class_{novelty_class}_norm_l2',
                                f'kfold_{N_FOLDS}_folds', 'vectors')
            else:
                windowed = True
                datapath = os.path.join(LOFAR_FOLDER, 
                                f'lofar_dataset_fft_pts_{fft_pts}_overlap_{OVERLAP}_decimation_{decimation}_pca_{PCA}_novelty_class_{novelty_class}_norm_l2',
                                f'kfold_{N_FOLDS}_folds', f'window_size_{WINDOW_SIZE}_stride_{STRIDE}')
                                
            print(LINES + f'fft_pts: {fft_pts}\n decimation: {decimation}\nModel: {model_name}\nNeurons: {num_neurons}\nNovelty class: {novelty_class}\n' + LINES)
            nn_evaluation.get_val_results_per_novelty(current_dir, datapath, model_name, THRESHOLD, CLASSES_NAMES, nov_index, windowed)

            current_dir, _ = os.path.split(current_dir)
            
        current_dir, _ = os.path.split(current_dir)
    
    current_dir, _ = os.path.split(current_dir)