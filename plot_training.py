"""This script plots the training metrics progression for each fold and for all models"""

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

def accuracy_plot(model_name, metrics_dict, metric_name, cache_dir):
    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    higher_epoch = 0

    for fold, val_metric in zip(range(1,len(metrics_dict['val_' + metric_name])+1), model_metrics['val_' + metric_name]):
        epochs = len(val_metric)
        if epochs > higher_epoch:
            higher_epoch = epochs
        plt.plot(np.arange(1,epochs+1), val_metric, label=f'fold {fold} training')

    for fold, fit_metric in zip(range(1,len(metrics_dict[metric_name])+1), metrics_dict[metric_name]):
        plt.plot(np.arange(1,len(fit_metric)+1), fit_metric, label=f'fold {fold} training')
                    
    plt.title(f'{model_name.captalize()} training progression')
    plt.ylim(0,1)
    plt.xlim(1, higher_epoch)
    plt.legend(loc=4, fontsize='x-small')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(cache_dir, f'{model_name}_training_progression.png'), format='png', dpi=200)
    plt.close()

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
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
            expert_model = True
        elif splitted_name == 'old':    #Avoids old data
            current_dir, _ = os.path.split(current_dir)
            continue
        else:
            model_name =splitted_name[0]
            expert_model = False

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            folds = np.sort(os.listdir(current_dir))
            novelty_class = method_novelty_folder[-1]

            create_dict = True
            
            for fold in folds: #Here we work on each fold speratedly, inside each fold folder
                current_dir = os.path.join(current_dir, fold)

                #Obatining data from each fold

                if expert_model:
                    with open(os.path.join(current_dir, 'committee_training_log.json'), 'r') as json_file:
                        committee_log = json.load(json_file)
                    
                    with open(os.path.join(current_dir, 'model_training_log.json'), 'r') as json_file:
                        wrapper_log = json.load(json_file)
                    
                    if create_dict:
                        expert_metrics = {class_ : {metric: list() for metric in committee_log[class_]['params']['metrics']} 
                                            for class_ in committee_log.keys()}
                        wrapper_metrics = {metric: list() for metric in wrapper_log['params']['metrics']}
                        create_dict = False
                    
                    for class_ in expert_metrics.keys():
                        for metric in expert_metrics[class_].keys():
                            expert_metrics[class_][metric].append(committee_log[class_]['history'][metric])
                    
                    for metric in wrapper_metrics.keys():
                        wrapper_metrics[metric].append(wrapper_log['history'][metric])

                else:
                    with open(os.path.join(current_dir, 'model_training_log.json'), 'r') as json_file:
                        fit_log = json.load(json_file)
                    
                    if create_dict:
                        model_metrics = {metric: list() for metric in fit_log['params']['metrics']}
                        create_dict = False                        
                    
                    for metric in fit_log['params']['metrics']:
                            model_metrics[metric].append(fit_log['history'][metric])

                current_dir, _ = os.path.split(current_dir)
            
            if expert_model:
                for class_, metrics_dict in expert_metrics.items():
                    accuracy_plot(f'{model_name}_{class_}_expert', metrics_dict, 'expert_accuracy', current_dir)

                accuracy_plot('wrapper_' + model_name, wrapper_metrics, 'sparse_accuracy', current_dir)
            else:
                accuracy_plot(model_name, model_metrics, 'sparse_accuracy', current_dir)
            
            current_dir, _ = os.path.split(current_dir)
    
        current_dir, _ = os.path.split(current_dir)
    
    current_dir, _ = os.path.split(current_dir)