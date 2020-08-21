import os
import gc
import json
from itertools import product

import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import data_analysis.utils.utils as da_utils
from data_analysis.neural_networks.training import MultiInitLog
from data_analysis.utils.lofar_operators import LofarImgSequence, LofarSequence
from data_analysis.utils.math_utils import trapezoid_integration
from data_analysis.model_evaluation import novelty_analysis
from neural_networks import models, utils

def get_avg_err(data):
    avg = np.sum(data, axis=0)/len(data)
    err = np.sqrt(np.var(data, axis=0))
    for i in range(len(err)):
        err[i], decimals = da_utils.around(err[i])
        decimals = 3 if err[i] == 0 else decimals
        avg[i] = np.around(avg[i], decimals)
    return avg, err

def output_report(output, classf, labels, classes_neurons, filepath=None):
    """Returns a report from the neurons output where the columns are the results from the neurons of the classes
    and the rows are the true labels of that class. on the cm column the columns are the predicted classes"""
    data = list()
    num_neurons = output.shape[-1]
    classes = list(classes_neurons.keys())
    have_zeros = [False for _ in range(num_neurons)]

    outer_column = ['all_data', 'all_data_err', 'max_5', 'max_5_err', 'min_5', 'min_5_err', 'cm']
    columns = pd.MultiIndex.from_product([outer_column, classes])
    columns = columns.insert(0, ('n_events', 'n_events'))
    
    for class_, neuron in classes_neurons.items():

        class_output = output[labels == class_]
        n_events = len(class_output)
        if n_events == 0:
            have_zeros[neuron] = True
            zeros_array = np.zeros(len(columns)-len(classes), dtype = np.uint8)
            data.append(zeros_array)
            continue

        class_output_avg, class_output_err = get_avg_err(class_output)
        sortarg = class_output[:, neuron].argsort(axis=0)
        max_5_percent = class_output[sortarg[np.math.floor(len(sortarg)*0.95):]]
        max_5_avg, max_5_err = get_avg_err(max_5_percent)
        min_5_percent = class_output[sortarg[:np.math.ceil(len(sortarg)*0.05)]]
        min_5_avg, min_5_err = get_avg_err(min_5_percent)
        to_append = np.concatenate(([n_events], class_output_avg, class_output_err, max_5_avg, max_5_err, min_5_avg, min_5_err), axis=0)        
        data.append(to_append)

    cm = confusion_matrix(labels, classf, normalize='true')
    
    if np.any(have_zeros):
        current_size = len(cm)
        for i in range(len(have_zeros)):
            if have_zeros[i]:
                cm = np.insert(cm, i, np.zeros(current_size, dtype=np.uint8), axis=1)
                current_size += 1
                cm = np.insert(cm, i, np.zeros(current_size, dtype=np.uint8), axis=0)

    data = np.concatenate((np.array(data), cm), axis=1)
    index = classes
    report_frame = pd.DataFrame(data, columns=columns, index=index)

    if not filepath is None:
        if not filepath is None:
            folder, _ = os.path.split(filepath)
            if not os.path.exists(folder):
                os.makedirs(folder)
            report_frame.to_csv(filepath, index = False)

    return report_frame

def plot_cm(cm, classes, cmerr = None, filepath=None, 
            xlabel='Predicted label', ylabel='True label', title='Confusion Matrix'):

    n_classes = len(classes)

    fig, axis = plt.subplots()
    im = axis.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    thresh = (cm.max() + cm.min())/2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i,j] < thresh else cmap_min
        if cmerr is None:
            text = str(np.around(cm[i,j], 3))
        else:
            err, decimals = da_utils.around(cmerr[i,j])
            value = np.around(cm[i,j], decimals)
            text = fr"{value} $\pm$ {err}"
        axis.text(j, i, text, ha='center', va='center', color=color)
    
    axis.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=classes,
            yticklabels=classes,
            ylabel=ylabel,
            xlabel=xlabel,
            title=title)
    
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(filepath, dpi=200, format='png')
    return fig, axis

def plot_rep(rep, output_path, name_prefix=''):
    try:
        rep['cm_err']
        avg_frame=True
        final_str = '_avg_plot.png'
    except KeyError:
        avg_frame=False
        final_str = '_plot.png'

    columns = ('all_data', 'max_5', 'min_5', 'cm')

    for column in columns:
        filename = os.path.join(output_path, name_prefix + column + final_str)
        if (not avg_frame) and (column == 'cm'):
            plot_cm(rep.loc[:,column].values, 
                    np.array(rep.loc[:,column].columns, dtype=str),
                    filepath = filename)
        else:
            title_name = ' '.join(column.split('_'))
            if title_name[-1] == '5':
                title_name = title_name + ' percent'
            if title_name == 'cm':
                plot_cm(rep.loc[:,column].values, 
                        np.array(rep.loc[:,column].columns, dtype=str),
                        rep.loc[:,column + '_err'].values,
                        filename)
            else:
                plot_cm(rep.loc[:,column].values, 
                        np.array(rep.loc[:,column].columns, dtype=str),
                        rep.loc[:,column + '_err'].values,
                        filename,
                        xlabel='Class neuron',
                        title='Output avg ' + title_name)
        plt.close()

def plot_training(metric_name, model_name, novelty_class, n_folds, model_metrics, output_path):
    
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

def frame_avg_var(frames_list, output_path, name_prefix=''):
    frames_avg = sum(frames_list)/len(frames_list)
    frames_avg.to_csv(os.path.join(output_path, name_prefix + 'frames_avg.csv'))
    values = np.array([frame.values for frame in frames_list])
    variances = np.array([np.var(values[:,i,:], axis=0) for i in range(values.shape[1])])
    frames_var = pd.DataFrame(data=variances, index=frames_avg.index, columns=frames_avg.columns)
    frames_var.to_csv(os.path.join(output_path, name_prefix + 'frames_var.csv'))

    return frames_avg, frames_var

def output_report_avg(reps, output_path, name_prefix=''):
    
    avg_frame = sum(reps)/len(reps)

    classes = np.array(avg_frame.loc[:,'cm'].columns.values, dtype=str)
    cm_err = np.sqrt(np.var([rep.loc[:,'cm'].values for rep in reps], axis=0))
    cm_err = pd.DataFrame(cm_err, columns=pd.MultiIndex.from_product([['cm_err'], classes]), index=classes)
    avg_frame = avg_frame.join(cm_err)
    del cm_err

    errors = np.array([rep.loc[:, 'all_data_err'].values for rep in reps])
    avg_frame.loc[:, 'all_data_err'] = np.sqrt(np.sum(errors ** 2, axis=0))/len(errors)
    errors = np.array([rep.loc[:, 'min_5_err'].values for rep in reps])
    avg_frame.loc[:, 'min_5_err'] = np.sqrt(np.sum(errors ** 2, axis=0))/len(errors)
    errors = np.array([rep.loc[:, 'max_5_err'].values for rep in reps])
    avg_frame.loc[:, 'max_5_err'] = np.sqrt(np.sum(errors ** 2, axis=0))/len(errors)

    events_err = np.sqrt(np.var([rep.loc[:,'n_events'].values for rep in reps], axis=0))
    avg_frame.insert(1, ('n_events', 'err'), events_err)

    avg_frame.to_csv(os.path.join(output_path, name_prefix + 'rep_avg.csv'))

    return avg_frame

def eval_fold(novelty_class, results_path, output_path, name_prefix=''):
    results_frame = pd.read_csv(results_path, header=[0,1])
    
    output = results_frame.loc[:,'Neurons'].values
    labels = np.array(results_frame.loc[:, ('Labels', 'L')].values.flatten(), dtype=str)
    classf = np.array(results_frame.iloc[:, 3].values, dtype=str)
    classes = np.unique(classf)
    classes_neurons = {class_: neuron for class_, neuron in zip(classes, range(len(classes)))}

    classf_rep = output_report(output[labels != 'Nov'], classf[labels != 'Nov'], labels[labels != 'Nov'], classes_neurons,
                                os.path.join(output_path, name_prefix + 'classf_rep.csv'))
    nov_rep = output_report(output[labels == 'Nov'], classf[labels == 'Nov'], classf[labels == 'Nov'], classes_neurons,
                                os.path.join(output_path, name_prefix + 'nov_rep.csv'))
    evaluation_frame = novelty_analysis.evaluate_nov_detection(results_frame, 
                                                filepath=os.path.join(output_path, name_prefix + 'eval_frame.csv'))

    plot_rep(nov_rep, output_path, name_prefix + 'nov_')
    plot_rep(classf_rep, output_path, name_prefix + 'classf_')
    novelty_analysis.plot_noc_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'noc_curve.png'))
    novelty_analysis.plot_accuracy_curve(results_frame, novelty_class, filepath=os.path.join(output_path, name_prefix + 'acc_curve.png'))

    return nov_rep, classf_rep, evaluation_frame

def eval_all_folds(nov_reps, classf_reps, eval_frames, output_path, name_prefix=''):

    classf_rep_avg = output_report_avg(classf_reps, output_path, name_prefix + 'classf_')
    nov_rep_avg = output_report_avg(nov_reps, output_path, name_prefix + 'nov_')

    eval_frames_avg, eval_frames_var = frame_avg_var(eval_frames, output_path, name_prefix + 'eval_')

    noc_areas = np.array([trapezoid_integration(frame.loc['Nov rate'].values.flatten(), frame.loc['Trigger rate'].values.flatten())
                            for frame in eval_frames])
    noc_area_avg = np.sum(noc_areas)/len(noc_areas)
    noc_area_var = np.var(noc_areas)
    
    with open(os.path.join(output_path, name_prefix + 'noc_area_dict.json'), 'w') as json_file:
        noc_area_dict = dict(folds=noc_areas, avg=noc_area_avg, var=noc_area_var)
        json.dump(da_utils.cast_to_python(noc_area_dict), json_file, indent=4)

    return nov_rep_avg, classf_rep_avg, eval_frames_avg, eval_frames_var, noc_area_dict

def evaluate_kfolds(current_dir, model_name, folds, novelty_class):

    eval_frames = list()
    classf_reps = list()
    nov_reps = list()

    create_dict = True
    fold_count = 1

    if model_name.split('_')[-1] == 'expert':
        is_expert=True
        expert_metrics = dict()
    else:
        is_expert=False

    for fold in folds: #Here we work on each fold speratedly, inside each fold folder
        current_dir = os.path.join(current_dir, fold)

        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue

        print(f'Fold {fold_count} novelty {novelty_class}')
        if is_expert:
            current_dir = os.path.join(current_dir, 'experts_multi_init_log')
            for expert_log in np.sort(os.listdir(current_dir)):
                expert_name = '_'.join(expert_log.split('_')[:2])

                multi_init = MultiInitLog.from_json(os.path.join(current_dir, expert_log))
                best_init = multi_init.get_best_init('val_expert_accuracy', 'max', False)
                training_log = multi_init.inits[best_init]

                if create_dict:
                    model_metrics = {metric: list() for metric in training_log['params']['metrics']}
                    
                for metric in training_log['params']['metrics']:
                    model_metrics[metric].append(training_log['logs'][metric])
                
                expert_metrics[expert_name] = model_metrics
            
                del multi_init, best_init, training_log
            
            current_dir, _ = os.path.split(current_dir)

            if create_dict:
                create_dict = False
            
        else:
            multi_init = MultiInitLog.from_json(os.path.join(current_dir, 'multi_init_log.json'))
            best_init = multi_init.get_best_init('val_sparse_accuracy', 'max', False)
            training_log = multi_init.inits[best_init]
                
            if create_dict:
                model_metrics = {metric: list() for metric in training_log['params']['metrics']}
                create_dict = False                        
            
            for metric in training_log['params']['metrics']:
                    model_metrics[metric].append(training_log['logs'][metric])
            
            del multi_init, best_init, training_log
        nov_rep, classf_rep, eval_frame = eval_fold(novelty_class, os.path.join(current_dir, 'results_frame.csv'), current_dir)
        eval_frames.append(eval_frame)
        classf_reps.append(classf_rep)
        nov_reps.append(nov_rep)

        fold_count += 1
        current_dir, _ = os.path.split(current_dir)

    print(f'Computing on all folds. Novelty {novelty_class}')
    if is_expert:
        for expert_name, metrics_log in expert_metrics.items():
            plot_training('expert_accuracy', expert_name, novelty_class, fold_count, metrics_log, current_dir)
            plot_training('loss', expert_name, novelty_class, fold_count, metrics_log , current_dir)
    else:
        plot_training('sparse_accuracy', model_name, novelty_class, fold_count, model_metrics, current_dir)
        plot_training('loss', model_name, novelty_class, fold_count, model_metrics, current_dir)

    threshold = np.array(eval_frame.columns.values.flatten(), dtype=np.float64)
    nov_rep_avg, classf_rep_avg, eval_frames_avg, eval_frames_var, noc_area_dict = eval_all_folds(nov_reps, classf_reps, eval_frames, current_dir)

    # Confusion matrices
    plot_rep(nov_rep_avg, current_dir, 'nov_')
    plot_rep(classf_rep_avg, current_dir, 'classf_')

    #Average Noc curve
    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.plot(eval_frames_avg.loc['Trigger rate'], eval_frames_avg.loc['Nov rate'], color='k')
    x = np.linspace(0,1, 10000)
    plt.plot(x, 1-x, color='k', linestyle='--')
    plt.title(f'Novelty class {novelty_class}')
    plt.ylabel('Average Trigger Rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel('Average Novelty Rate')
    plt.fill_between(eval_frames_avg.loc['Nov rate'], eval_frames_avg.loc['Trigger rate'], interpolate=True, color='#808080')
    noc_area_err, decimals = da_utils.around(np.sqrt(noc_area_dict['var']))
    noc_area_avg = round(noc_area_dict['avg'], decimals)
    plt.text(0.3, 0.25, fr"Area=({noc_area_avg} $\pm$ {noc_area_err})",
                horizontalalignment='center', fontsize=20)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(current_dir, 'average_noc_curve.png'), dpi=200, format='png')
    plt.close()

    #All folds noc curve
    plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.title(f'Novelty class {novelty_class}')
    zip_iterator = zip(noc_area_dict['folds'], eval_frames, range(1, fold_count))
    for area, eval_frame, fold in zip_iterator:
        plt.plot(eval_frame.loc['Nov rate'], eval_frame.loc['Trigger rate'], 
                    label=f'Fold {fold} Area = {round(area, 2)}')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.ylabel('Trigger Rate')
    plt.xlabel('Novelty Rate')
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(current_dir, 'noc_curve_all_folds.png'), dpi=200, format='png')
    plt.close()

    #Average acc curves
    fig = plt.figure(figsize=(12,3))
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.ylim(0,1)
    plt.xlim(-1,1)
    errorbar = plt.errorbar(threshold, eval_frames_avg.loc['Nov acc'], yerr=np.sqrt(eval_frames_var.loc['Nov acc']), label='Nov acc', errorevery=20)
    nov_acc_color = errorbar.lines[0].get_color()
    errorbar = plt.errorbar(threshold, eval_frames_avg.loc['Classf acc'], yerr=np.sqrt(eval_frames_var.loc['Classf acc']), label='Classf acc', errorevery=20)
    classf_acc_color = errorbar.lines[0].get_color()
    plt.title(f'Novelty class {novelty_class}')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Threshold')
    plt.legend(loc=4)

    zoomed_axis = fig.add_axes([0.17,0.22,0.3,0.3])
    zoomed_axis.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    zoomed_axis.errorbar(threshold, eval_frames_avg.loc['Nov acc'], yerr=np.sqrt(eval_frames_var.loc['Nov acc']), label='Nov acc', errorevery=20, color=nov_acc_color)
    zoomed_axis.errorbar(threshold, eval_frames_avg.loc['Classf acc'], yerr=np.sqrt(eval_frames_var.loc['Classf acc']), label='Classf acc', errorevery=20, color=classf_acc_color)
    zoomed_axis.set_ylim(0.5,1.0)
    zoomed_axis.set_xlim(0.5,1)
    zoomed_axis.set_yticks([0.5, 0.625, 0.75, 0.875, 1.0])

    fig.savefig(fname=os.path.join(current_dir, 'average_acc_curve.png'), dpi=200, format='png')
    plt.close(fig)

def get_val_results_per_novelty(output_dir, datapath, model_name, threshold, classes_names, nov_index, windowed):

    folds = np.sort(os.listdir(output_dir))
    split = 0
    for fold in folds:

        current_dir = os.path.join(output_dir, fold)
        if not os.path.isdir(current_dir):
            current_dir, _ = os.path.split(current_dir)
            continue
        print(f'At fold {split}')
        datapath = os.path.join(datapath, f'split_{split}.npz')
        data = np.load(datapath, allow_pickle=True)
        model_architecture = getattr(models, model_name)

        if models.expert_commitee in model_architecture.__bases__:
            model = utils.load_expert_committee(os.path.join(current_dir, 'experts_multi_init_log'), 'val_expert_accuracy', 'max', False)
        else:
            log = utils.MultiInitLog.from_json(os.path.join(current_dir, 'multi_init_log.json'))
            model = log.get_best_model(metric='val_sparse_accuracy', mode='max', training_end=False)

        if windowed:
            val_set = LofarImgSequence(data['data'], data['x_val'])
        else:
            val_set = LofarSequence(data['x_val'])

        predictions = model.predict(val_set)
        novelty_analysis.get_results(predictions=predictions, labels=data['y_val'], threshold=threshold, 
                                        classes_names=classes_names, novelty_index=nov_index,
                                        filepath=os.path.join(current_dir, 'val_results_frame.csv'))

        del model, predictions
        
        gc.collect()
        keras.backend.clear_session()

        split += 1
        current_dir, _ = os.path.split(current_dir)
        datapath, _ = os.path.split(datapath)