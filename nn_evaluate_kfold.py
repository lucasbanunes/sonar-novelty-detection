"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import gc
import multiprocessing
import argparse

import numpy as np

from config import setup
#Gets enviroment variables and appends needed custom packages
setup()
from neural_networks.evaluation import evaluate_kfolds

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
        del splitted, splitted_name

        if model_name[:3] == 'old':
            current_dir, _ = os.path.split(current_dir)
            continue
        
        processes = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds (folder with all the folds)
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))

            novelty_class = method_novelty_folder[-1]
            fold_count = 1
            processes.append(multiprocessing.Process(target=evaluate_kfolds, args=(current_dir, model_name, folds, novelty_class), daemon=True))

            current_dir, _ = os.path.split(current_dir)
        
        if __name__ == '__main__':
            print(LINES + f'fft_pts: {fft_pts}\n decimation: {decimation}\nModel: {model_name}\nNeurons: {num_neurons}\n' + LINES)
            for p in processes:
                p.start()
            
            for p in processes:
                p.join()

            for p in processes:
                p.close()
            
            del processes
            gc.collect()
            
        current_dir, _ = os.path.split(current_dir)
    
    current_dir, _ = os.path.split(current_dir)