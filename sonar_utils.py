import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_analysis.utils.utils as da_utils
from data_analysis.utils import math_utils
from data_analysis.model_evaluation import novelty_analysis

def save_multi_init_log(log, output_dir):
    callback = log.pop('best_callback')
    new_log = {'params': callback.params, 'history': callback.history, 'inits_log': log['inits_log']}
    new_log = da_utils.cast_to_python(new_log)
    with open(os.path.join(output_dir, 'model_training_log.json'), 'w') as json_file:
        json.dump(new_log, json_file, indent=4)

def save_expert_log(experts_logs, output_dir):
    expert_json = {class_: {'params': expert_log['best_callback'].params, 'history': expert_log['best_callback'].history} for class_, expert_log in experts_logs.items()}
    expert_json = da_utils.cast_to_python(expert_json)
    with open(os.path.join(output_dir, 'committee_training_log.json'), 'w') as json_file:
        json.dump(expert_json, json_file, indent=4)

def get_results_message(multi_init_log):
    message_values = dict()
    message_values['best_init'] = multi_init_log['inits_log']['best_init']
    for metric in multi_init_log['best_callback'].params['metrics']:
        message_values[metric] = multi_init_log['best_callback'].history[metric][-1]
    introduction = 'Results of the previous training were:\n'
    results = '\n'.join([f'{metric_name}: {metric_value}' for metric_name, metric_value in message_values.items()])
    return introduction + results

def eval_results(classes_names, threshold, novelty_class, nov_index, 
                    predictions, labels, output_path, name_prefix=''):
    results_frame = novelty_analysis.get_results(predictions=predictions, labels=labels, threshold=threshold, 
                                                classes_names=classes_names, novelty_index=nov_index,
                                                filepath=os.path.join(output_path, name_prefix + 'results_frame.csv'))

    evaluation_frame = novelty_analysis.evaluate_nov_detection(results_frame, 
                                                filepath=os.path.join(output_path, name_prefix + 'eval_frame.csv'))

    novelty_analysis.plot_noc_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'noc_curve.png'))
    novelty_analysis.plot_accuracy_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'acc_curve.png'))

    return evaluation_frame

def folds_eval(eval_frames, output_path, name_prefix=''):
        eval_frames_avg = sum(eval_frames)/len(eval_frames)
        eval_frames_avg.to_csv(os.path.join(output_path, name_prefix + 'eval_frames_avg.csv'))
        metrics = np.array([frame.values for frame in eval_frames])
        variances = np.array([np.var(metrics[:,i,:], axis=0) for i in range(metrics.shape[1])])
        eval_frames_var = pd.DataFrame(data=variances, index=eval_frames_avg.index, columns=eval_frames_avg.columns)
        eval_frames_var.to_csv(os.path.join(output_path, name_prefix + 'eval_frames_var.csv'))

        noc_areas = np.array([math_utils.trapezoid_integration(frame.loc['Nov rate'].values.flatten(), frame.loc['Trigger rate'].values.flatten())
                                for frame in eval_frames])
        noc_area_avg = np.sum(noc_areas)/len(noc_areas)
        noc_area_var = np.var(noc_areas)
        
        with open(os.path.join(output_path, name_prefix + 'noc_area_dict.json'), 'w') as json_file:
            noc_area_dict = dict(folds=noc_areas, avg=noc_area_avg, var=noc_area_var)
            json.dump(da_utils.cast_to_python(noc_area_dict), json_file, indent=4)

        return eval_frames_avg, eval_frames_var, noc_area_dict

def plot_data(novelty_class, folds, colors, threshold,
                eval_frames, eval_frame_avg, eval_frame_var, noc_area_dict, output_path, name_prefix=''):

        #Average Noc curve
        plt.figure(figsize=(12,3))
        plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.plot(eval_frame_avg.loc['Trigger rate'], eval_frame_avg.loc['Nov rate'], color='k')
        x = np.linspace(0,1, 10000)
        plt.plot(x, 1-x, color='k', linestyle='--')
        plt.title(f'Novelty class {novelty_class}')
        plt.ylabel('Average Trigger Rate')
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.xlabel('Average Novelty Rate')
        plt.fill_between(eval_frame_avg.loc['Nov rate'], eval_frame_avg.loc['Trigger rate'], interpolate=True, color='#808080')
        noc_area_err, decimals = da_utils.around(np.sqrt(noc_area_dict['var']))
        noc_area_avg = round(noc_area_dict['avg'], decimals)
        plt.text(0.3, 0.25, fr"Area=({noc_area_avg} $\pm$ {noc_area_err})",
                    horizontalalignment='center', fontsize=20)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(output_path, name_prefix + 'average_noc_curve.png'), dpi=200, format='png')
        plt.close()

        #All folds noc curve
        plt.figure(figsize=(12,3))
        plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.title(f'Novelty class {novelty_class}')
        zip_iterator = zip(noc_area_dict['folds'], eval_frames, colors[:folds], range(1, folds+1))
        for area, eval_frame, color , fold in zip_iterator:
            plt.plot(eval_frame.loc['Nov rate'], eval_frame.loc['Trigger rate'], 
                        label=f'Fold {fold} Area = {round(area, 2)}', color=color)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.ylabel('Trigger Rate')
        plt.xlabel('Novelty Rate')
        plt.legend(loc=3)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(output_path, name_prefix + 'noc_curve_all_folds.png'), dpi=200, format='png')
        plt.close()

        #Average acc curves
        fig = plt.figure(figsize=(12,3))
        plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.ylim(0,1)
        plt.xlim(-1,1)
        errorbar = plt.errorbar(threshold, eval_frame_avg.loc['Nov acc'], yerr=np.sqrt(eval_frame_var.loc['Nov acc']), label='Nov acc', errorevery=20)
        nov_acc_color = errorbar.lines[0].get_color()
        errorbar = plt.errorbar(threshold, eval_frame_avg.loc['Classf acc'], yerr=np.sqrt(eval_frame_var.loc['Classf acc']), label='Classf acc', errorevery=20)
        classf_acc_color = errorbar.lines[0].get_color()
        plt.title(f'Novelty class {novelty_class}')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Threshold')
        plt.legend(loc=4)

        zoomed_axis = fig.add_axes([0.1,0.3,0.3,0.3])
        zoomed_axis.errorbar(threshold, eval_frame_avg.loc['Nov acc'], yerr=np.sqrt(eval_frame_var.loc['Nov acc']), label='Nov acc', errorevery=20, color=nov_acc_color)
        zoomed_axis.errorbar(threshold, eval_frame_avg.loc['Classf acc'], yerr=np.sqrt(eval_frame_var.loc['Classf acc']), label='Classf acc', errorevery=20, color=classf_acc_color)
        zoomed_axis.set_ylim(0.5,1.0)
        zoomed_axis.set_xlim(0.5,1)
        zoomed_axis.set_yticks([0.5, 0.625, 0.75, 0.875, 1.0])

        fig.savefig(fname=os.path.join(output_path, name_prefix + 'average_acc_curve.png'), dpi=200, format='png')
        plt.close(fig)

def evaluate_kfolds_committee(current_dir, folds, novelty_class, colors):

    exp_eval_frames = list()
    wrapper_eval_frames = list()
    fold_count = 1

    for fold in folds: #Here we work on each fold speratedly, inside each fold folder
        current_dir = os.path.join(current_dir, fold)

        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        
        print(f'Fold {fold_count}')
        exp_results_frame = pd.read_csv(os.path.join(current_dir, 'committee_results_frame.csv'), header=[0,1])
        exp_results_frame.to_csv(os.path.join(current_dir, 'exp_results_frame.csv'))
        exp_eval_frame = novelty_analysis.evaluate_nov_detection(exp_results_frame, 
                                                                    filepath=os.path.join(current_dir, 'exp_eval_frame.csv'))
        novelty_analysis.plot_noc_curve(exp_results_frame, novelty_class, filepath=os.path.join(current_dir, 'exp_noc_curve.png'))
        novelty_analysis.plot_accuracy_curve(exp_results_frame, novelty_class, filepath=os.path.join(current_dir, 'exp_acc_curve.png'))
        exp_eval_frames.append(exp_eval_frame)

        wrapper_results_frame = pd.read_csv(os.path.join(current_dir, 'results_frame.csv'), header=[0,1])
        wrapper_results_frame.to_csv(os.path.join(current_dir, 'wrapper_results_frame.csv'))
        wrapper_eval_frame = novelty_analysis.evaluate_nov_detection(wrapper_results_frame, 
                                                                    filepath=os.path.join(current_dir, 'wrapper_eval_frame.csv'))
        novelty_analysis.plot_noc_curve(wrapper_results_frame, novelty_class, filepath=os.path.join(current_dir, 'wrapper_noc_curve.png'))
        novelty_analysis.plot_accuracy_curve(wrapper_results_frame, novelty_class, filepath=os.path.join(current_dir, 'wrapper_acc_curve.png'))
        wrapper_eval_frames.append(wrapper_eval_frame)

        fold_count += 1
        current_dir, _ = os.path.split(current_dir)

    print('Averaging all folds.')

    exp_threshold = np.array(exp_eval_frame.columns.values.flatten(), dtype=np.float64)
    exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict = folds_eval(exp_eval_frames, current_dir, 'exp_')
    plot_data(novelty_class, fold_count, colors, exp_threshold,
                            exp_eval_frames, exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict,
                            current_dir, 'exp_')

    wrapper_threshold = np.array(wrapper_eval_frame.columns.values.flatten(), dtype=np.float64)
    wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict = folds_eval(wrapper_eval_frames, current_dir, 'wrapper_')
    plot_data(novelty_class, fold_count, colors, wrapper_threshold,
                            wrapper_eval_frames, wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict,
                            current_dir, 'wrapper_')

def evaluate_kfolds_standard(current_dir, folds, novelty_class, colors):

    eval_frames = list()
    fold_count = 1

    for fold in folds: #Here we work on each fold speratedly, inside each fold folder
        current_dir = os.path.join(current_dir, fold)

        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue

        print(f'Fold {fold_count}')
        results_frame = pd.read_csv(os.path.join(current_dir, 'results_frame.csv'), header=[0,1])
        eval_frame = novelty_analysis.evaluate_nov_detection(results_frame, 
                                                                filepath=os.path.join(current_dir, 'eval_frame.csv'))

        novelty_analysis.plot_noc_curve(results_frame, novelty_class, filepath=os.path.join(current_dir, 'noc_curve.png'))
        novelty_analysis.plot_accuracy_curve(results_frame, novelty_class, filepath=os.path.join(current_dir, 'acc_curve.png'))
        eval_frames.append(eval_frame)

        fold_count += 1
        current_dir, _ = os.path.split(current_dir)

    print('Averaging all folds.')

    threshold = np.array(eval_frame.columns.values.flatten(), dtype=np.float64)
    eval_frames_avg, eval_frames_var, noc_area_dict = folds_eval(eval_frames, current_dir)
    plot_data(novelty_class, fold_count, colors, threshold,
                            eval_frames, eval_frames_avg, eval_frames_var, noc_area_dict,
                            current_dir)


class SendInitMessage:

    def __init__(self, bot, chat_id, prior_message=None):
        self.bot  = bot
        self.chat_id = chat_id
        if prior_message is None:
            self.prior_message = ''

    def __call__(self, init, *args, **kwargs):
        self.bot.sendMessage(self.chat_id, self.prior_message + f'Finished init {init}')