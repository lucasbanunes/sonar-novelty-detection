"""This script plots the training metrics progression for each fold and for all models"""


import os
import gc
import multiprocessing
import numpy as np
import pandas as pd
from setup import setup

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()
OUTPUT_DIR += '_test'
import sonar_utils

LINES = '--------------------------------------------------------------------\n'

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    if not os.path.isdir(current_dir):
        continue

    fft_pts = lofar_params_folder.split('_')[2]
    decimation = lofar_params_folder.split('_')[-2]

    model_neurons_folders = np.sort(os.listdir(current_dir))

    for model_neurons_folder in model_neurons_folders:
        current_dir = os.path.join(current_dir, model_neurons_folder)
        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        method_novelty_folders = np.sort(os.listdir(current_dir))
        
        num_neurons = model_neurons_folder[-2:]
        splitted_name = model_neurons_folder.split('_')

        if 'old' in splitted_name:
            current_dir, _ = os.path.split(current_dir)
            del splitted_name
            continue
        else:
            model_name = '_'.join(splitted_name[0:-2])
            del splitted_name

        if model_name.split('_')[-1] == 'expert':
            function = sonar_utils.plot_committee_training
        else:
            function = sonar_utils.plot_standard_training
        
        processes = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds (folder with all the folds)
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))
            novelty_class = method_novelty_folder[-1]
            fold_count = 1

            processes.append(multiprocessing.Process(target=function, args=(current_dir, folds, model_name, novelty_class), daemon=True))

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