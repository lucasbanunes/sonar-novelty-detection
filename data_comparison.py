"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

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

def round_value_error(value, error):
    error, decimals = utils.around(error)
    value = round(value, decimals)
    return value, error

def round_as_array(arr):
    arr[0], arr[1] = round_value_error(arr[0], arr[1])
    return arr

lofar_params_folders = np.sort(os.listdir(OUTPUT_DIR))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(OUTPUT_DIR, lofar_params_folder)
    if not os.path.isdir(current_dir):
        continue
    model_neurons_folders = np.sort(os.listdir(current_dir))

    outer_index = list()
    inner_index = list()
    classes = ['A', 'B', 'C', 'D']
    metrics = ['Média', 'Erro']
    data = list()

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

        outer_index.append(model_name.capitalize())
        inner_index.append(num_neurons)
        data_per_novelty = list()

        #If there are experts to be proccessed
        experts_processing = False
        expert_data_per_novelty = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))

            print(f'Getting data in {current_dir}')

            try:
                with open(os.path.join(current_dir, 'metrics_info.json'), 'r') as json_file:
                    metrics_info = json.load(json_file)
            except Exception:
                break
            
            noc_areas = np.array([value for key, value in metrics_info['noc_area'].items() if key.split('_')[0] == 'fold'])
            noc_var = np.var(noc_areas, axis=0)
            metrics_info['noc_area']['var'] = float(noc_var)
            error, decimals = utils.around(np.sqrt(noc_var))
            avg = round(metrics_info['noc_area']['avg'], decimals)
            data_per_novelty.append(avg)
            data_per_novelty.append(error)

            novelty_class = method_novelty_folder[-1]
            novelty_index = list(CLASSES_NAMES).index(novelty_class)
            experts_trig = list()
            experts_nov_rate = list()
            experts_acc = list()
            fold_count=0

            with open(os.path.join(current_dir, 'metrics_info.json'), 'w') as json_file:
                json.dump(metrics_info, json_file, indent=4)
            
            del metrics_info
            
            for fold in folds: #Here we work on each fold speratedly, inside each fold folder
                current_dir = os.path.join(current_dir, fold)

                if not os.path.isdir(current_dir):
                    current_dir, _ = os.path.split(current_dir)
                    continue

                try:    #Tests if there are experts to analyze and applt wta on their output
                    experts_frame = pd.read_csv(os.path.join(current_dir, 'experts_frame.csv'), index_col=0)
                    experts_predictions = experts_frame.values
                    results_frame = pd.read_csv(os.path.join(current_dir, 'results_frame.csv'), header=[0,1])
                    threshold = np.array(results_frame.loc[:,'Classification'].columns.values, dtype=np.float64)
                    labels = results_frame.loc[:, 'Labels'].values.flatten()
                    labels = np.where(labels == 'Nov', novelty_class, labels)
                    for label, class_ in zip(range(4), CLASSES_NAMES):
                        labels = np.where(labels == class_, label, labels)
                    labels = np.array(labels, dtype=np.int32)
                    experts_results_frame = novelty_analysis.get_results(predictions=experts_predictions, labels=labels, threshold=threshold, 
                                                classes_names=CLASSES_NAMES, novelty_index=novelty_index,
                                                filepath=os.path.join(current_dir, 'experts_results_frame.csv'))

                    y_true = np.where(experts_results_frame.loc[:, 'Labels'].values.flatten() == 'Nov', 1, 0)     #Novelty is 1, known data is 0
                    novelty_matrix = np.where(experts_results_frame.loc[:,'Classification'] == 'Nov', 1, 0)
                    experts_acc.append(np.apply_along_axis(lambda x: accuracy_score(y_true, x), 0, novelty_matrix))
                    trig, nov_rate = np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None), 0, novelty_matrix)
                    experts_trig.append(trig)
                    experts_nov_rate.append(nov_rate)
                    experts_processing = True
                    fold_count += 1

                except FileNotFoundError:
                    current_dir, _ = os.path.split(current_dir)
                    break                        

                current_dir, _ = os.path.split(current_dir)

            #Implementar o nome e os valores corretos
            if experts_processing:
                experts_trig = np.array(experts_trig)
                experts_nov_rate = np.array(experts_nov_rate)
                trigger_avg = np.sum(experts_trig, axis=0)/len(experts_trig)
                trigger_std = np.std(experts_trig, axis=0)
                novelty_avg = np.sum(experts_nov_rate, axis=0)/len(experts_nov_rate)
                novelty_std = np.std(experts_nov_rate, axis=0)
                noc_areas = np.array([utils.NumericalIntegration.trapezoid_rule(nov_rate, trig) for nov_rate, trig in zip(experts_nov_rate, experts_trig)])
                noc_area_avg = np.sum(noc_areas)/len(noc_areas)
                noc_area_std = np.std(noc_areas)
                experts_acc = np.array(experts_acc)
                acc_avg = np.sum(experts_acc, axis=0)/len(experts_acc)
                acc_std = np.std(experts_acc, axis=0)

                noc_area_dict = {f'fold_{fold}': noc_area for fold, noc_area in zip(range(1,fold_count+1), noc_areas)}
                noc_area_dict['avg'] = noc_area_avg
                noc_area_dict['std'] = noc_area_std

                novelty_rate_dict = {f'fold_{fold}': nov_rate for fold, nov_rate in zip(range(1,fold_count+1), experts_nov_rate)}
                novelty_rate_dict['avg'] = novelty_avg
                novelty_rate_dict['std'] = novelty_std

                trigger_dict = {f'fold_{fold}': trig for fold, trig in zip(range(1,fold_count+1), experts_trig)}
                trigger_dict['avg'] = trigger_avg
                trigger_dict['std'] = trigger_std

                acc_dict={f'fold_{fold}': accuracy for fold, accuracy in zip(range(1,fold_count+1), experts_acc)}
                acc_dict['avg'] = acc_avg
                acc_dict['std'] = acc_std


                with open(os.path.join(current_dir, 'experts_metrics_info.json'), 'w') as json_file:
                    metrics_info = dict(threshold=threshold, acc=acc_dict, trigger=trigger_dict, novelty_rate=novelty_rate_dict, noc_area=noc_area_dict)
                    json.dump(utils.cast_to_python(metrics_info), json_file, indent=4)

                std, decimals = utils.around(metrics_info['noc_area']['std'])
                avg = round(metrics_info['noc_area']['avg'], decimals)
                expert_data_per_novelty.append(avg)
                expert_data_per_novelty.append(std)
            
            current_dir, _ = os.path.split(current_dir)
    
        data.append(data_per_novelty)

        if experts_processing:
            outer_index.append(model_name.capitalize())
            inner_index.append(f'{num_neurons}_experts')
            data.append(expert_data_per_novelty)
            experts_processing=False

        current_dir, _ = os.path.split(current_dir)

    data = np.array(data)
    #import pdb; pdb.set_trace()

    index = pd.MultiIndex.from_arrays([outer_index, inner_index], names=('Model', 'Neurons'))
    columns = pd.MultiIndex.from_product([classes, metrics])
    try:
        comparison_frame = pd.DataFrame(data=data, index=index, columns=columns)
    except Exception:
        continue

    avg_values = list()
    error_values = list()
    for column in comparison_frame:
        if column[-1] == 'Média':
            avg_values.append(comparison_frame.loc[:,column].values)
        elif column[-1] == 'Erro':
            error_values.append(comparison_frame.loc[:,column].values)
        else:
            raise ValueError

    avg_values = np.stack(tuple(avg_values), axis=-1)
    error_values = np.stack(tuple(error_values), axis=-1)

    num_occurences = avg_values.shape[-1]
    novelties_avg = np.sum(avg_values, axis=1)/num_occurences
    novelties_error = (1/num_occurences)*(np.sqrt(np.sum(error_values**2, axis=1)))

    data = np.stack((novelties_avg, novelties_error), axis=-1)
    data = np.apply_along_axis(round_as_array, 1, data)
    columns = pd.MultiIndex.from_product([['Novidade'],['Média', 'Erro']])
    novelties_avg_frame = pd.DataFrame(data, columns = columns, index = comparison_frame.index)
    comparison_frame = comparison_frame.join(novelties_avg_frame)
    comparison_frame.to_csv(os.path.join(current_dir, 'comparison_frame.csv'))
    
    current_dir, _ = os.path.split(current_dir)