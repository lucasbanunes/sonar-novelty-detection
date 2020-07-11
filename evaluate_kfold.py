"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import gc
import multiprocessing
import numpy as np
import pandas as pd
from setup import setup

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()
#OUTPUT_DIR = 'D:\\Workspace\\output\\old_sonar_08-05-20'

import sonar_utils
from data_analysis.model_evaluation import novelty_analysis

COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
LINES = '--------------------------------------------------------------------\n'

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    #import pdb; pdb.set_trace()
    if not os.path.isdir(current_dir):
        continue

    fft_pts = lofar_params_folder.split('_')[2]
    decimation = lofar_params_folder.split('_')[-2]
    
    if fft_pts == '1024':
        continue

    model_neurons_folders = np.sort(os.listdir(current_dir))

    outer_index = list()
    inner_index = list()
    data = list()

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
            function = sonar_utils.evaluate_kfolds_committee
        else:
            function = sonar_utils.evaluate_kfolds_standard
        
        processes = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds (folder with all the folds)
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))
            novelty_class = method_novelty_folder[-1]
            fold_count = 1

            processes.append(multiprocessing.Process(target=function, args=(current_dir, folds, novelty_class, COLORS), daemon=True))

            current_dir, _ = os.path.split(current_dir)
        
        if __name__ == '__main__':
            print(LINES + f'fft_pts: {fft_pts}\n decimation: {decimation}\nModel: {model_name}\nNeurons: {num_neurons}')
            for p in processes:
                p.start()
            
            for p in processes:
                p.join()
            
            del processes
            gc.collect()
            
        current_dir, _ = os.path.split(current_dir)
    
    current_dir, _ = os.path.split(current_dir)