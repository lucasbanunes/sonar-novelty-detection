import os
import gc
import sys
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.metrics import recall_score, accuracy_score

os_name = platform.system()

if os_name == 'Linux':
    sys.path.append('/home/lucas_nunes/Repositories/data-analysis')
    LOFAR_FOLDER = '/home/lucas_nunes/Documentos/datasets/lofar_data'
    OUTPUT_DIR = '/home/lucas_nunes/Documentos/sonar_output'
    RAW_DATA = '/home/lucas_nunes/Documentos/datasets/runs_info'
elif os_name == 'Windows':
    sys.path.append('D:\\Repositories\\data-analysis')
    LOFAR_FOLDER = 'D:\\datasets\\lofar_data'
    OUTPUT_DIR = 'D:\\Workspace\\output\\sonar'
    RAW_DATA = 'D:\\datasets\\runs_info'
else:
    raise ValueError(f'{os_name} is not supported.')

from data_analysis.model_evaluation import novelty_analysis
from data_analysis.utils import utils

def get_avg_std(*values):
    output = list()
    for value in values:
        value = np.array(value)
        output.append(np.sum(value, axis=0)/len(value))
        output.append(np.std(value, axis=0))
    return output

def kfold_noc_curves(trigger, novelty_rate, fold_count, novelty_class, output_dir):

    COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
    if fold_count>len(COLORS):
        raise ValueError(f'the maximum number of colors in this script for plotting is {len(COLORS)} please add more')

    trigger = np.array(trigger)
    novelty_rate = np.array(novelty_rate)
    #noc_areas = np.array([utils.NumericalIntegration.trapezoid_rule(nov_rate, trig) for nov_rate, trig in zip(novelty_rate, trigger)])
    #trigger_avg, trigger_std, novelty_avg, novelty_std, noc_area_avg, noc_area_std = get_avg_std(trigger, novelty_rate, noc_areas)
    trigger_avg = np.sum(trigger, axis=0)/len(trigger)
    novelty_avg = np.sum(novelty_rate, axis=0)/len(novelty_rate)
    noc_areas = np.array([utils.NumericalIntegration.trapezoid_rule(nov_rate, trig) for nov_rate, trig in zip(novelty_rate, trigger)])
    noc_area_avg = np.sum(noc_areas)/len(noc_areas)
    noc_area_std = np.std(noc_areas)

    #Noc curves
    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.plot(trigger_avg, novelty_avg, color='k')
    x = np.linspace(0,1, 10000)
    plt.plot(x, 1-x, color='k', linestyle='--')
    plt.title(f'Novelty class {novelty_class}')
    plt.ylabel('Average Trigger Rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('Average Novelty Rate')
    plt.fill_between(novelty_avg, trigger_avg, interpolate=True, color='#808080')
    noc_area_std, decimals = utils.around(noc_area_std)
    noc_area_avg = round(noc_area_avg, decimals)
    plt.text(0.3, 0.25, fr"Area=({noc_area_avg} $\pm$ {noc_area_std})",
                horizontalalignment='center', fontsize=20)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(output_dir, 'average_noc_curve.png'), dpi=200, format='png')
    plt.close()

    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.title(f'Novelty class {novelty_class}')
    for area, fold_trig, fold_nov_rate, color , fold in zip(noc_areas, trigger, novelty_rate, COLORS[:fold_count-1], range(1, fold_count+1)):
        plt.plot(fold_nov_rate, fold_trig, label=f'Fold {fold} Area = {round(area, 2)}', color=color)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.ylabel('Trigger Rate')
    plt.xlabel('Novelty Rate')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(output_dir, 'noc_curve_all_folds.png'), dpi=200, format='png')
    plt.close()