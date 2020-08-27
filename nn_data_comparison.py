"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import gc
import json
import argparse

import pandas as pd
import numpy as np

from config import setup
#Gets enviroment variables and appends needed custom packages
setup()
from data_analysis.utils import utils
LINES = '--------------------------------------------------------------------\n'

parser = argparse.ArgumentParser(description='Evaluates the results of a neural network training',
                                    prog='Neual Network Kfold evaluation')
parser.add_argument('dir', help='Directory generated from nn_kfold.py with the results and training history')
parser.add_argument('--format', help='Format to read and save the data frame: csv or xlsx.\nDefaults to xlsx', default='csv', type=str, choices=['csv', 'xlsx'])
args = parser.parse_args()
output_dir = args.dir
file_format = args.format

CLASSES_NAMES = ('A', 'B', 'C', 'D')
METRICS = ('Average', 'Error')
FILE_NAME = 'noc_comparison_frame'

def round_value_error(value, error):
    error, decimals = utils.around(error)
    value = round(value, decimals)
    return value, error

def round_as_array(arr):
    arr[0], arr[1] = round_value_error(arr[0], arr[1])
    return arr

lofar_params_folders = np.sort(os.listdir(output_dir))
for lofar_params_folder in lofar_params_folders:
    current_dir = os.path.join(output_dir, lofar_params_folder)
    if not os.path.isdir(current_dir):
        continue
    
    splitted = lofar_params_folder.split('_')
    fft_pts = splitted[2]
    decimation = splitted[5]
    pca = splitted[-3]
    bins = splitted[-1]

    outer_index = list()
    inner_index = list()
    data = list()

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
        
        data_per_novelty = list()
        processes = list()

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds (folder with all the folds)
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            print(f'Getting data in {current_dir}')
            novelty_class = method_novelty_folder[-1]
            fold_count = 1

            with open(os.path.join(current_dir, 'noc_area_dict.json'), 'r') as json_file:
                noc_area_dict = json.load(json_file)

            error, decimals = utils.around(np.sqrt(noc_area_dict['var']))
            avg = round(noc_area_dict['avg'], decimals)
            data_per_novelty.append(avg)
            data_per_novelty.append(error)

            current_dir, _ = os.path.split(current_dir)

        data.append(data_per_novelty)
        outer_index.append(model_name.capitalize())
        inner_index.append(num_neurons)

        current_dir, _ = os.path.split(current_dir)
    
    index = pd.MultiIndex.from_arrays([outer_index, inner_index], names=('Model', 'Neurons'))
    columns = pd.MultiIndex.from_product([CLASSES_NAMES, METRICS])
    novelties_comparison_frame = pd.DataFrame(data=data, index=index, columns=columns)

    avg_values = list()
    error_values = list()
    for column in novelties_comparison_frame:
        if column[-1] == 'Average':
            avg_values.append(novelties_comparison_frame.loc[:,column].values)
        elif column[-1] == 'Error':
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
    columns = pd.MultiIndex.from_product([['Novelties'],['Average', 'Error']])
    novelties_avg_frame = pd.DataFrame(data, columns = columns, index = novelties_comparison_frame.index)

    comparison_frame = novelties_comparison_frame.join(novelties_avg_frame)
    if file_format == 'csv':
        comparison_frame.to_csv(os.path.join(current_dir, FILE_NAME+'.csv'))
    elif file_format == 'xlsx':
        comparison_frame.to_excel(os.path.join(current_dir, FILE_NAME+'.xlsx'))
    else:
        raise ValueError(f'Unknown format type: {file_format}')

    del comparison_frame, novelties_comparison_frame, novelties_avg_frame
    
    gc.collect()    
    current_dir, _ = os.path.split(current_dir)