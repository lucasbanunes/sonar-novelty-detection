"""This script mounts a DataFrame with the  NOC AUC results with all the models in a given lofar parameters folder"""

import os
import numpy as np
import pandas as pd
from setup import setup

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()

import sonar_utils
from data_analysis.model_evaluation import novelty_analysis

COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')

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

        for method_novelty_folder in method_novelty_folders: #Here We get all the folds, folder with all the folds
            current_dir = os.path.join(current_dir, method_novelty_folder)
            if not os.path.isdir(current_dir):
                current_dir, _ = os.path.split(current_dir)
                continue
            folds = np.sort(os.listdir(current_dir))
            novelty_class = method_novelty_folder[-1]
            fold_count = 1

            print(f'Getting data in {current_dir}')

            if model_name.split('_')[-1] == 'expert':
                experts_processing = True
                exp_eval_frames = list()
                wrapper_eval_frames = list()
            else:
                experts_processing = False
                eval_frames = list()
            
            for fold in folds: #Here we work on each fold speratedly, inside each fold folder
                current_dir = os.path.join(current_dir, fold)

                if not os.path.isdir(current_dir):
                    current_dir, _ = os.path.split(current_dir)
                    continue
                
                print(f'In fold {fold_count}')

                if experts_processing:
                    exp_results_frame = pd.read_csv(os.path.join(current_dir, 'committee_results_frame.csv'), header=[0,1])
                    exp_results_frame.to_csv(os.path.join(current_dir, 'exp_results_frame'), index=False)
                    exp_eval_frame = novelty_analysis.evaluate_nov_detection(exp_results_frame, 
                                                                                filepath=os.path.join(current_dir, 'exp_eval_frame.csv'))
                    novelty_analysis.plot_noc_curve(exp_results_frame, novelty_class, filepath=os.path.join(current_dir, 'exp_noc_curve.png'))
                    novelty_analysis.plot_accuracy_curve(exp_results_frame, novelty_class, filepath=os.path.join(current_dir, 'exp_acc_curve.png'))
                    exp_eval_frames.append(exp_eval_frame)

                    wrapper_results_frame = pd.read_csv(os.path.join(current_dir, 'committee_results_frame.csv'), header=[0,1])
                    wrapper_results_frame.to_csv(os.path.join(current_dir, 'wrapper_results_frame'), index=False)
                    wrapper_eval_frame = novelty_analysis.evaluate_nov_detection(wrapper_results_frame, 
                                                                                filepath=os.path.join(current_dir, 'wrapper_eval_frame.csv'))
                    novelty_analysis.plot_noc_curve(wrapper_results_frame, novelty_class, filepath=os.path.join(current_dir, 'wrapper_noc_curve.png'))
                    novelty_analysis.plot_accuracy_curve(wrapper_results_frame, novelty_class, filepath=os.path.join(current_dir, 'wrapper_acc_curve.png'))
                    wrapper_eval_frames.append(wrapper_eval_frame)
                else:
                    results_frame = pd.read_csv(os.path.join(current_dir, 'results_frame.csv'), header=[0,1])
                    eval_frame = novelty_analysis.evaluate_nov_detection(results_frame, 
                                                                            filepath=os.path.join(current_dir, 'eval_frame.csv'))

                    novelty_analysis.plot_noc_curve(results_frame, novelty_class, filepath=os.path.join(current_dir, 'noc_curve.png'))
                    novelty_analysis.plot_accuracy_curve(results_frame, novelty_class, filepath=os.path.join(current_dir, 'acc_curve.png'))
                    eval_frames.append(eval_frame)

                fold_count += 1
                current_dir, _ = os.path.split(current_dir)

            print('Averaging all folds.')

            if experts_processing:
                exp_threshold = exp_eval_frame.columns.values.flatten()
                exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict = sonar_utils.folds_eval(exp_eval_frames, current_dir)
                sonar_utils.plot_data(novelty_class, fold_count, COLORS, threshold,
                                        exp_eval_frames, exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict,
                                        current_dir)

                wrapper_threshold = wrapper_eval_frame.columns.values.flatten()
                wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict = sonar_utils.folds_eval(wrapper_eval_frames, current_dir)
                sonar_utils.plot_data(novelty_class, fold_count, COLORS, threshold,
                                        wrapper_eval_frames, wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict,
                                        current_dir)
            else:
                threshold = eval_frame.columns.values.flatten()
                eval_frames_avg, eval_frames_var, noc_area_dict = sonar_utils.folds_eval(eval_frames, current_dir)
                sonar_utils.plot_data(novelty_class, fold_count, COLORS, threshold,
                                        eval_frames, eval_frames_avg, eval_frames_var, noc_area_dict,
                                        current_dir)
 
            current_dir, _ = os.path.split(current_dir)
    
        current_dir, _ = os.path.split(current_dir)
    
    current_dir, _ = os.path.split(current_dir)