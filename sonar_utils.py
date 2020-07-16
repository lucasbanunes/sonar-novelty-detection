import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_analysis.utils.utils as da_utils
from data_analysis.utils import math_utils, metrics
from data_analysis.model_evaluation import novelty_analysis

def save_multi_init_log(log, filepath):
    callback = log.pop('best_callback')
    new_log = {'params': callback.params, 'history': callback.history, 'inits_log': log['inits_log']}
    new_log = da_utils.cast_to_python(new_log)
    with open(filepath, 'w') as json_file:
        json.dump(new_log, json_file, indent=4)

def save_expert_log(experts_logs, filepath):
    expert_json = {class_: {'params': expert_log['best_callback'].params, 'history': expert_log['best_callback'].history} for class_, expert_log in experts_logs.items()}
    expert_json = da_utils.cast_to_python(expert_json)
    with open(filepath, 'w') as json_file:
        json.dump(expert_json, json_file, indent=4)

def get_results_message(multi_init_log):
    message_values = dict()
    message_values['best_init'] = multi_init_log['inits_log']['best_init']
    for metric in multi_init_log['best_callback'].params['metrics']:
        message_values[metric] = multi_init_log['best_callback'].history[metric][-1]
    introduction = 'Results of the previous training were:\n'
    results = '\n'.join([f'{metric_name}: {metric_value}' for metric_name, metric_value in message_values.items()])
    return introduction + results

def eval_results(novelty_class, results_path, output_path, name_prefix=''):
    results_frame = pd.read_csv(results_path, header=[0,1])
    
    labels = np.array(results_frame.loc[:, ('Labels', 'L')].values.flatten(), dtype=str)

    classf_pred = np.array(results_frame.iloc[:, 3].values, dtype=str)
    classf_pred = classf_pred[labels != 'Nov']
    classf_true = labels[labels != 'Nov']
    cm_frame = metrics.confusion_matrix_frame(classf_true, classf_pred, filepath = os.path.join(output_path, name_prefix + 'cm_frame.csv'), normalize='true')
    metrics.plot_confusion_matrix(cm_frame, os.path.join(output_path, name_prefix + 'cm_plot.png'))

    evaluation_frame = novelty_analysis.evaluate_nov_detection(results_frame, 
                                                filepath=os.path.join(output_path, name_prefix + 'eval_frame.csv'))

    novelty_analysis.plot_noc_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'noc_curve.png'))
    novelty_analysis.plot_accuracy_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'acc_curve.png'))

    return cm_frame, evaluation_frame

def frame_avg_var(frames_list, output_path, name_prefix=''):
    frames_avg = sum(frames_list)/len(frames_list)
    frames_avg.to_csv(os.path.join(output_path, name_prefix + 'frames_avg.csv'))
    values = np.array([frame.values for frame in frames_list])
    variances = np.array([np.var(values[:,i,:], axis=0) for i in range(values.shape[1])])
    frames_var = pd.DataFrame(data=variances, index=frames_avg.index, columns=frames_avg.columns)
    frames_var.to_csv(os.path.join(output_path, name_prefix + 'frames_var.csv'))

    return frames_avg, frames_var

def folds_eval(cm_frames, eval_frames, output_path, name_prefix=''):

        cm_frames_avg, cm_frames_var = frame_avg_var(cm_frames, output_path, name_prefix + 'cm_')
        eval_frames_avg, eval_frames_var = frame_avg_var(eval_frames, output_path, name_prefix + 'eval_')

        noc_areas = np.array([math_utils.trapezoid_integration(frame.loc['Nov rate'].values.flatten(), frame.loc['Trigger rate'].values.flatten())
                                for frame in eval_frames])
        noc_area_avg = np.sum(noc_areas)/len(noc_areas)
        noc_area_var = np.var(noc_areas)
        
        with open(os.path.join(output_path, name_prefix + 'noc_area_dict.json'), 'w') as json_file:
            noc_area_dict = dict(folds=noc_areas, avg=noc_area_avg, var=noc_area_var)
            json.dump(da_utils.cast_to_python(noc_area_dict), json_file, indent=4)

        return cm_frames_avg, cm_frames_var, eval_frames_avg, eval_frames_var, noc_area_dict

def plot_data(novelty_class, folds, colors, threshold,
                cm_frames_avg, cm_frames_var,
                eval_frames, eval_frame_avg, eval_frame_var, noc_area_dict, 
                output_path, name_prefix=''):

        # Confusion matrix plot
        metrics.plot_confusion_matrix(cm_frames_avg, cmerr=np.sqrt(cm_frames_var.values), 
                                filepath=os.path.join(output_path, name_prefix + 'cm_frames_avg_plot.png'))

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

        zoomed_axis = fig.add_axes([0.17,0.22,0.3,0.3])
        zoomed_axis.errorbar(threshold, eval_frame_avg.loc['Nov acc'], yerr=np.sqrt(eval_frame_var.loc['Nov acc']), label='Nov acc', errorevery=20, color=nov_acc_color)
        zoomed_axis.errorbar(threshold, eval_frame_avg.loc['Classf acc'], yerr=np.sqrt(eval_frame_var.loc['Classf acc']), label='Classf acc', errorevery=20, color=classf_acc_color)
        zoomed_axis.set_ylim(0.5,1.0)
        zoomed_axis.set_xlim(0.5,1)
        zoomed_axis.set_yticks([0.5, 0.625, 0.75, 0.875, 1.0])

        fig.savefig(fname=os.path.join(output_path, name_prefix + 'average_acc_curve.png'), dpi=200, format='png')
        plt.close(fig)

def evaluate_kfolds_committee(current_dir, model_name, folds, novelty_class, colors):

    exp_eval_frames = list()
    exp_cm_frames = list()

    wrapper_eval_frames = list()
    wrapper_cm_frames = list()
    create_dict = True    

    fold_count = 1

    for fold in folds: #Here we work on each fold speratedly, inside each fold folder
        current_dir = os.path.join(current_dir, fold)

        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        
        print(f'Fold {fold_count} novelty {novelty_class}')

        with open(os.path.join(current_dir, 'exp_training_log.json'), 'r') as json_file:
            exp_log = json.load(json_file)
        
        if create_dict:
            exp_metrics = {class_ : {metric: list() for metric in log['params']['metrics']} for class_, log in exp_log.items()}
        
        for class_, log in exp_log.items():
            for metric in log['params'] ['metrics']:
                exp_metrics[class_][metric].append(log['history'][metric])

        with open(os.path.join(current_dir, 'wrapper_training_log.json'), 'r') as json_file:
            wrapper_log = json.load(json_file)
        
        if create_dict:
            wrapper_metrics = {metric: list() for metric in wrapper_log['params']['metrics']}
            create_dict = False
        
        for metric in wrapper_log['params']['metrics']:
                wrapper_metrics[metric].append(wrapper_log['history'][metric])
        
        del wrapper_log, exp_log

        exp_cm_frame, exp_eval_frame = eval_results(novelty_class, os.path.join(current_dir, 'exp_results_frame.csv'), current_dir, 'exp_')
        exp_eval_frames.append(exp_eval_frame)
        exp_cm_frames.append(exp_cm_frame)

        wrapper_cm_frame, wrapper_eval_frame = eval_results(novelty_class, os.path.join(current_dir, 'wrapper_results_frame.csv'), current_dir, 'wrapper_')
        wrapper_eval_frames.append(wrapper_eval_frame)
        wrapper_cm_frames.append(wrapper_cm_frame)


        fold_count += 1
        current_dir, _ = os.path.split(current_dir)

    print(f'Computing all folds. Novelty {novelty_class}')

    training_plot('sparse_accuracy', 'wrapper_'+model_name, novelty_class, fold_count, wrapper_metrics, current_dir)
    training_plot('loss', 'wrapper_'+model_name, novelty_class, fold_count, wrapper_metrics, current_dir)

    for class_, metrics_log in exp_metrics.items():
        training_plot('expert_accuracy', f'{class_}_exp_'+model_name, novelty_class, fold_count, metrics_log, current_dir)
        training_plot('loss', f'{class_}_exp_'+model_name, novelty_class, fold_count, metrics_log , current_dir)

    exp_threshold = np.array(exp_eval_frame.columns.values.flatten(), dtype=np.float64)
    exp_cm_frames_avg, exp_cm_frames_var, exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict = folds_eval(exp_cm_frames, 
                                                                                                                    exp_eval_frames, 
                                                                                                                    current_dir, 'exp_')
    plot_data(novelty_class, fold_count, colors, exp_threshold,
                exp_cm_frames_avg, exp_cm_frames_var,
                exp_eval_frames, exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict,
                current_dir, 'exp_')

    wrapper_threshold = np.array(wrapper_eval_frame.columns.values.flatten(), dtype=np.float64)
    wrapper_cm_frames_avg, wrapper_cm_frames_var, wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict = folds_eval(wrapper_cm_frames, 
                                                                                                                    wrapper_eval_frames, 
                                                                                                                    current_dir, 'wrapper_')
    plot_data(novelty_class, fold_count, colors, wrapper_threshold,
                wrapper_cm_frames_avg, wrapper_cm_frames_var,
                wrapper_eval_frames, wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_area_dict,
                current_dir, 'wrapper_')

def evaluate_kfolds_standard(current_dir, model_name, folds, novelty_class, colors):

    eval_frames = list()
    cm_frames = list()
    create_dict = True
    fold_count = 1

    for fold in folds: #Here we work on each fold speratedly, inside each fold folder
        current_dir = os.path.join(current_dir, fold)

        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue

        print(f'Fold {fold_count} novelty {novelty_class}')

        with open(os.path.join(current_dir, 'training_log.json'), 'r') as json_file:
            training_log = json.load(json_file)
        
        if create_dict:
            model_metrics = {metric: list() for metric in training_log['params']['metrics']}
            create_dict = False                        
        
        for metric in training_log['params']['metrics']:
                model_metrics[metric].append(training_log['history'][metric])
        
        del training_log

        cm_frame, eval_frame = eval_results(novelty_class, os.path.join(current_dir, 'results_frame.csv'), current_dir)
        eval_frames.append(eval_frame)
        cm_frames.append(cm_frame)


        fold_count += 1
        current_dir, _ = os.path.split(current_dir)

    print(f'Computing on all folds. Novelty {novelty_class}')

    training_plot('sparse_accuracy', model_name, novelty_class, fold_count, model_metrics, current_dir)
    training_plot('loss', model_name, novelty_class, fold_count, model_metrics, current_dir)

    threshold = np.array(eval_frame.columns.values.flatten(), dtype=np.float64)
    cm_frames_avg, cm_frames_var, eval_frames_avg, eval_frames_var, noc_area_dict = folds_eval(cm_frames, eval_frames, current_dir)
    plot_data(novelty_class, fold_count, colors, threshold,
                cm_frames_avg, cm_frames_var,
                eval_frames, eval_frames_avg, eval_frames_var, noc_area_dict,
                current_dir)

def training_plot(metric_name, model_name, novelty_class, n_folds, model_metrics, output_path):
    
    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    higher_epoch = 0

    for fold, val_metric, train_metric in zip(range(1, n_folds), model_metrics['val_'+metric_name], model_metrics[metric_name]):
        current_epochs = len(val_metric)
        higher_epoch = current_epochs if current_epochs > higher_epoch else higher_epoch
        plt.plot(np.arange(1, current_epochs+1), val_metric, label=f'fold {fold} val')
        plt.plot(np.arange(1, current_epochs+1), train_metric, label=f'fold {fold} train')

    plt.title(f'{model_name.capitalize()} novelty class {novelty_class} {metric_name} progression')
    plt.xlim(1, higher_epoch)
    plt.legend(loc=4, fontsize='x-small')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{model_name}_{metric_name}__progression.png'), format='png', dpi=200)
    plt.close()

class SendInitMessage:

    def __init__(self, bot, chat_id, prior_message=None):
        self.bot  = bot
        self.chat_id = chat_id
        if prior_message is None:
            self.prior_message = ''

    def __call__(self, init, *args, **kwargs):
        self.bot.sendMessage(self.chat_id, self.prior_message + f'Finished init {init}')