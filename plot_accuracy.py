"""This script reads metrics_info.json and classf_metrics.json obtains the accuracy for both classification and novelty detection and plots them
per threshold in same axis. This action is made for each novelty class and for an average between the classes."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from setup import setup
from copy import deepcopy

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()

import my_models
from data_analysis.model_evaluation import novelty_analysis
from data_analysis.utils import utils

COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])

def plot_data(fig, axis, threshold, nov_acc_avg, classf_acc_avg, nov_acc_err, class_acc_err):
    axis.errorbar(threshold, nov_acc_avg, yerr=nov_acc_err, label='Detecção', color='red', errorevery=15)
    axis.errorbar(threshold, classf_acc_avg, yerr=classf_acc_err, label='Classificação', color='blue', errorevery=15)
    axis.legend(loc=4)
    zoomed_axis = fig.add_axes([0.2,0.2,0.4,0.2])
    selection_index = np.where(threshold == 0.5)[0][0]
    zoomed_axis.errorbar(threshold[selection_index:], nov_acc_avg[selection_index:], yerr=nov_acc_err[selection_index:], label='Detecção', color='red', errorevery=15)
    zoomed_axis.errorbar(threshold[selection_index:], classf_acc_avg[selection_index:], yerr=classf_acc_err[selection_index:], label='Classificação', color='blue', errorevery=15)
    zoomed_axis.set_ylim(0.5,1.0)
    zoomed_axis.set_yticks([0.5, 0.75, 1.0])
    return fig, axis

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    #import pdb;pdb.set_trace()
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    if not os.path.isdir(current_dir):
        print('Continuing in loop lofar_params_folder')
        print(current_dir)
        current_dir, _ = os.path.split(current_dir)
        continue
    model_neurons_folders = np.sort(os.listdir(current_dir))

    lofar_params_name = lofar_params_folder.split('_')
    fft_pts = int(lofar_params_name[2])
    decimation = int(lofar_params_name[-2])

    for model_neurons_folder in model_neurons_folders:
        current_dir = os.path.join(current_dir, model_neurons_folder)
        if not os.path.isdir(current_dir):
            print('Continuing in loop model_neurons_folder')
            print(current_dir)
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

        all_novelties = {'nov':list(), 'classf':list()}

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                print('Continuing in method novelty_folder')
                print(current_dir)
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))
            novelty_class = method_novelty_folder[-1]
            print(f'At {fft_pts} fft_pts {decimation} decimation {model_name} model with {num_neurons} neurons and novelty class {novelty_class}')
            print(current_dir)

            with open(os.path.join(current_dir, 'classf_metrics.json'), 'r') as json_file:
                classf_metrics = json.load(json_file)
                classf_acc_err = np.sqrt(classf_metrics['acc']['var'])
                classf_acc_avg = np.array(classf_metrics['acc']['avg'])
                del classf_metrics
            with open(os.path.join(current_dir, 'metrics_info.json'), 'r') as json_file:
                metrics_info = json.load(json_file)
                acc = np.array([value for key, value in metrics_info['acc'].items() if key.split('_')[0] == 'fold'])
                nov_acc_err = np.sqrt(np.var(np.array(acc), axis=0))
                nov_acc_avg = np.array(metrics_info['acc']['avg'])
                threshold = np.array(metrics_info['threshold'])
                del metrics_info

            fig, axis = plt.subplots()
            axis.set_title(f'Acurácia de detecção e classificação para classe {novelty_class} novidade')
            axis.set_ylabel('Acurácia')
            axis.set_xlabel('Limiar')
            plot_data(fig, axis, threshold, nov_acc_avg, classf_acc_avg, nov_acc_err, classf_acc_err)
            plt.tight_layout()
            print(f"Saved fig at {os.path.join(current_dir, 'acc_plot.png')}")
            fig.savefig(os.path.join(current_dir, 'acc_plot.png'), dpi=200, format='png')
            plt.close(fig)

            #import pdb;pdb.set_trace()

            all_novelties['classf'].append(classf_acc_avg)
            all_novelties['nov'].append(nov_acc_avg)
            current_dir, _ = os.path.split(current_dir)
            
        nov_acc = np.array(all_novelties['nov'])
        classf_acc = np.array(all_novelties['classf'])

        del all_novelties

        nov_avg = np.sum(nov_acc, axis=0)/len(nov_acc)
        nov_var = np.var(nov_acc, axis=0)
        classf_avg = np.sum(classf_acc, axis=0)/len(nov_acc)
        classf_var = np.var(classf_acc, axis=0)

        fig, axis = plt.subplots()
        axis.set_title(f'Média de acurácia de detecção e classificação entre todas as novidades')
        axis.set_ylabel('Acurácia')
        axis.set_xlabel('Limiar')
        plot_data(fig, axis, threshold, nov_acc_avg, classf_acc_avg, nov_acc_err, classf_acc_err)
        plt.tight_layout()
        fig.savefig(os.path.join(current_dir, 'acc_plot_all_novelties.png'), dpi=200, format='png')
        plt.close(fig)

        with open(os.path.join(current_dir, 'all_novelties_metrics.json'), 'w') as json_file:
            metrics = dict()
            metrics['nov'] = {'acc':{'avg': nov_avg, 'var': nov_var}}
            metrics['classf'] = {'acc':{'avg': classf_avg, 'var': classf_var}}
            json.dump(utils.cast_to_python(metrics), json_file, indent=4)
    
        current_dir, _ = os.path.split(current_dir)

    current_dir, _ = os.path.split(current_dir)