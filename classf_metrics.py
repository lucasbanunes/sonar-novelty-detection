"""This script reads each results_frame.csv file from each fold and computes a given metrics for analysing the classification
instead of novelty, write a dict with the metrics in a json file"""

import os
import gc
import sys
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
from data_analysis.utils import utils

COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])

def class_acc_with_nov(y_pred, y_true):
    y_pred = y_pred[y_true != 'Nov']
    y_true = y_true[y_true != 'Nov']
    correct = np.sum(np.where(y_pred == y_true, 1, 0))
    total = len(y_true)
    return correct/total

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    if not os.path.isdir(current_dir):
        current_dir, _ = os.path.split(current_dir)
        continue
    model_neurons_folders = np.sort(os.listdir(current_dir))

    for model_neurons_folder in model_neurons_folders:
        current_dir = os.path.join(current_dir, model_neurons_folder)
        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        method_novelty_folders = np.sort(os.listdir(current_dir))

        num_neurons = model_neurons_folder[-2:]
        splitted_name = model_neurons_folder.split('_')
        if splitted_name[1] == 'expert':
            model_name = splitted_name[0] + '_' + splitted_name[1]
        elif splitted_name == 'old':    #Avoids old data
            current_dir, _ = os.path.split(current_dir)
            continue
        else:
            model_name =splitted_name[0]

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))

            novelty_class = method_novelty_folder[-1]
            classf_metrics = dict()
            acc_dict = dict()
            classf_acc = list()

            print(f'At {model_name} model with {num_neurons} neurons and novelty class {novelty_class}')
            
            for fold in folds: #Here we work on each fold speratedly, inside each fold folder
                current_dir = os.path.join(current_dir, fold)

                if not os.path.isdir(current_dir):
                    current_dir, _ = os.path.split(current_dir)
                    continue
                
                fold_count = current_dir[-1]
                results_frame = pd.read_csv(os.path.join(current_dir, 'results_frame.csv'), header=[0,1])
                y_true = results_frame.loc[:, 'Labels'].values.flatten()
                predictions = results_frame.loc[:, 'Classification'].values
                fold_classf_acc = np.apply_along_axis(lambda x: class_acc_with_nov(x, y_true), axis=0, arr=predictions)
                acc_dict[f'fold_{fold_count}'] = fold_classf_acc
                classf_acc.append(fold_classf_acc)
                del fold_count, results_frame, y_true, predictions, fold_classf_acc, 

                current_dir, _ = os.path.split(current_dir)
            
            classf_acc = np.array(classf_acc)
            acc_dict['avg'] = np.sum(classf_acc, axis=0)/len(classf_acc)
            acc_dict['std'] = np.std(classf_acc, axis=0)
            acc_dict['var'] = np.var(classf_acc, axis=0)
            acc_dict = utils.cast_to_python(acc_dict)
            classf_metrics['acc'] = acc_dict

            with open(os.path.join(current_dir, 'classf_metrics.json'), 'w') as json_file:
                json.dump(classf_metrics, json_file, indent=4)
            
            del classf_acc, acc_dict, classf_metrics
            
            current_dir, _ = os.path.split(current_dir)
    
        current_dir, _ = os.path.split(current_dir)

    current_dir, _ = os.path.split(current_dir)