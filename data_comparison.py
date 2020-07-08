"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
from setup import setup

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()
OUTPUT_DIR = OUTPUT_DIR + '_test'

from data_analysis.utils import utils

parser = argparse.ArgumentParser(description='This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder')
parser.add_argument('--format', help='Format to save the data frame: csv or xlsx.\nDefaults to xlsx', default='csv', type=str, choices=['csv', 'xlsx'])

FORMAT = parser.parse_args().format
COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
CLASSES_NAMES = ('A', 'B', 'C', 'D')
METRICS = ('Média', 'Erro')

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

        #If there are experts to be proccessed
        if model_name.split('_')[-1] == 'expert':
            experts_processing = True
            exp_data_per_novelty = list()
            wrapper_data_per_novelty = list()
        else:
            experts_processing = False
            data_per_novelty = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))

            print(f'Getting data in {current_dir}')
            if experts_processing:
                with open(os.path.join(current_dir, 'exp_noc_area_dict.json'), 'r') as json_file:
                    exp_noc_area_dict = json.load(json_file)

                exp_error, exp_decimals = utils.around(np.sqrt(exp_noc_area_dict['var']))
                exp_avg = round(exp_noc_area_dict['avg'], exp_decimals)
                exp_data_per_novelty.append(exp_avg)
                exp_data_per_novelty.append(exp_error)

                with open(os.path.join(current_dir, 'wrapper_noc_area_dict.json'), 'r') as json_file:
                    wrapper_noc_area_dict = json.load(json_file)

                wrapper_error, wrapper_decimals = utils.around(np.sqrt(wrapper_noc_area_dict['var']))
                wrapper_avg = round(wrapper_noc_area_dict['avg'], wrapper_decimals)
                wrapper_data_per_novelty.append(wrapper_avg)
                wrapper_data_per_novelty.append(wrapper_error)
            else:
                with open(os.path.join(current_dir, 'noc_area_dict.json'), 'r') as json_file:
                    noc_area_dict = json.load(json_file)

                error, decimals = utils.around(np.sqrt(noc_area_dict['var']))
                avg = round(noc_area_dict['avg'], decimals)
                data_per_novelty.append(avg)
                data_per_novelty.append(error)
            
            current_dir, _ = os.path.split(current_dir)
    
        if experts_processing:
            data.append(exp_data_per_novelty)
            outer_index.append(model_name.capitalize())
            inner_index.append(f'{num_neurons}_exp_wta')

            data.append(wrapper_data_per_novelty)
            outer_index.append(model_name.capitalize())
            inner_index.append(num_neurons)
        else:
            data.append(data_per_novelty)
            outer_index.append(model_name.capitalize())
            inner_index.append(num_neurons)

        current_dir, _ = os.path.split(current_dir)

    data = np.array(data)

    index = pd.MultiIndex.from_arrays([outer_index, inner_index], names=('Model', 'Neurons'))
    columns = pd.MultiIndex.from_product([CLASSES_NAMES, METRICS])
    novelties_comparison_frame = pd.DataFrame(data=data, index=index, columns=columns)

    avg_values = list()
    error_values = list()
    for column in novelties_comparison_frame:
        if column[-1] == 'Média':
            avg_values.append(novelties_comparison_frame.loc[:,column].values)
        elif column[-1] == 'Erro':
            error_values.append(novelties_comparison_frame.loc[:,column].values)
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
    novelties_avg_frame = pd.DataFrame(data, columns = columns, index = novelties_comparison_frame.index)

    comparison_frame = novelties_comparison_frame.join(novelties_avg_frame)
    if FORMAT == 'csv':
        comparison_frame.to_csv(os.path.join(current_dir, 'comparison_frame.csv'))
    elif FORMAT == 'xlsx':
        comparison_frame.to_excel(os.path.join(current_dir, 'comparison_frame.xlsx'))
    else:
        raise ValueError(f'Uknown format type: {FORMAT}')

    del comparison_frame, novelties_comparison_frame, novelties_avg_frame
    
    gc.collect()
    current_dir, _ = os.path.split(current_dir)