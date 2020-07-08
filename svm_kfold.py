import os
import gc
import time
import json
import joblib
import socket
import getpass
import argparse
import telegram
import collections
import threading
import numpy as np
from setup import setup
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()

import my_models
import utils as sonar_utils
from data_analysis.svm.models import OneVsRestSVCCommittee
from data_analysis.model_evaluation import novelty_analysis
from data_analysis.model_evaluation.splitters import LofarSplitter
import data_analysis.utils.utils as da_utils
from data_analysis.utils import lofar_operators

parser = argparse.ArgumentParser(description='Trains a SVM for novelty detection using one vs rest method and corss validates using kfold.',
                                    prog='SVC Committee Kfold validation')
parser.add_argument('fft_pts', help='number of fft points used to generate the lofar data', type=int)
parser.add_argument('overlap', help='number of overlap used to generate the lofar data', type=int)
parser.add_argument('decimation', help='number of decimation used to generate the lofar data', type=int)
parser.add_argument('novelty_class', help='class that will be treated as novelty')
parser.add_argument('folds', help='number of folds', type=int)
parser.add_argument('C', help='C parameter for regularization', type=float)
parser.add_argument('gamma', help='gamma paremeter for the rbf kernel', type=float)
parser.add_argument('pca_comp', help='number of PCA components', type=int)
parser.add_argument('--bot', help='telegram bot token to report', default=None)
parser.add_argument('--chat', help='chat id for the bot', default=None)

args = parser.parse_args()
fft_pts = args.fft_pts
overlap = args.overlap
decimation = args.decimation
novelty_class = args.novelty_class
folds = args.folds
c_param = args.C
gamma = args.gamma
pca_comp = args.pca_comp
TELEGRAM_BOT = args.telegram
CHAT_ID = args.chat

#CONSTANTS
#Contraints the maximum number of folds to 10
COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
if folds>len(COLORS):
    raise ValueError(f'the maximum number of colors in this script for plotting is {len(COLORS)} please add more')

#Constants
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])       #Name of the classes
CLASSES_VALUES = np.arange(4)      #Value of each class to be used in the model
CLASSES_DICT = {key:value for key, value in zip(CLASSES_NAMES, CLASSES_VALUES)}     #Dict conecting classes and its values

NOV_INDEX = list(CLASSES_NAMES).index(novelty_class)        #Index of the novelty class in CLASSES_NAMES
KNOWN_CLASSES_NAMES = np.delete(CLASSES_NAMES, NOV_INDEX)
KNOWN_CLASSES_VALUES = np.arange(3)
KNOWN_CLASSES_DICT = {known_class: value 
            for known_class, value in zip(KNOWN_CLASSES_NAMES, KNOWN_CLASSES_VALUES)}
            #Dict with the key as known names and value as known values

#Constants functions
VALUE_MAPPING = lambda x: CLASSES_NAMES[CLASSES_VALUES == x][0]        #Maps the values back to the classes names
FROM_FULL_TO_KNOWN = lambda x: KNOWN_CLASSES_DICT[VALUE_MAPPING(x)]       #Function that connects full values to known values
KNOWN_NAME_TO_KNOWN_VALUE = lambda x: KNOWN_CLASSES_DICT[x]        #Function that returns the known value of a known class


SVC_PARAMS = dict.fromkeys(KNOWN_CLASSES_NAMES, {'C': c_param, 'gamma': gamma, 'cache_size': 500, 'verbose':False})   #Params of the SVC's

#Getting a bot to give live updates on output
if TELEGRAM_BOT:
    MESSAGE_ID = f'{socket.gethostname()} at {getpass.getuser()} user.\n'
    bot = telegram.Bot(TELEGRAM_BOT)

#HERE THE SCRIPT REALLY STARTS
start_time = time.time()
folds_time = dict()

try:
    print(f'---------------------------- {fft_pts} FFT_PTS {decimation} DECIMATION ----------------------------')
    data, labels, freq = joblib.load(os.path.join(LOFAR_FOLDER, f'lofar_data_file_fft_{fft_pts}_overlap_{overlap}_decimation_{decimation}_spectrum_left_400.npy'))
    runs_per_class = joblib.load(os.path.join(RAW_DATA, f'runs_info_{fft_pts}_fft_pts_{decimation}_decimation.joblib'))

    data = lofar_operators.normalize(data, np.concatenate(tuple(runs_per_class.values()), axis=0), 2)
    pca = PCA(pca_comp, copy=False)
    data = pca.fit_transform(data)
    splitter = LofarSplitter(lofar_data=data, labels=labels, 
                            runs_per_classes=runs_per_class, classes_values_map=CLASSES_DICT)
    splitter.compile()

    output_dir = os.path.join(OUTPUT_DIR, f'lofar_parameters_{fft_pts}_fft_pts_{decimation}_decimation_{pca_comp}_pca_comp')
    output_dir = os.path.join(output_dir, f'svm_ovr_C_{c_param}_gamma_{gamma}', f'kfold_novelty_class_{novelty_class}')

    splitter.set_novelty(novelty_class, FROM_FULL_TO_KNOWN)

    fold_count = 1
    metrics_per_fold = list()
    params = 'Model: SVM ovr\n' + '\n'.join([f'{key}: {value}' for key,value in args.__dict__.items()])

    if TELEGRAM_BOT:
        message = MESSAGE_ID +'Starting the SVM kfold split with the given parameters\n' + params
        bot.sendMessage(CHAT_ID, message)

    for x_test, y_test, x_val, y_val, x_train, y_train in splitter.kfold_split(n_splits=folds, shuffle=True):

        fold_start = time.time()

        print('------------------------------ Control variables ------------------------------')
        print(f'current fold: {fold_count}')
        print(params)
        print('-------------------------------------------------------------------------------')

        output_dir = os.path.join(output_dir, f'fold_{fold_count}')

        if TELEGRAM_BOT:
            bot.sendMessage(CHAT_ID, MESSAGE_ID + f'Starting the SVM fitting for novelty_class {novelty_class} at fold {fold_count}')

        x_train = np.concatenate((x_train, x_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

        svc_committee = OneVsRestSVCCommittee(KNOWN_CLASSES_DICT, SVC_PARAMS)
        svc_committee.fit(x_train, y_train, verbose=True)
        predictions = svc_committee.predict(x_test)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        np.save(os.path.join(output_dir, 'x_test.npy'), x_test)        
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)        
        np.save(os.path.join(output_dir, 'x_val.npy'), x_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)           
        
        #Getting results and plotting the curve
        results_frame = novelty_analysis.svm_get_results(predictions=predictions, labels=y_test, 
                                                        classes_names=KNOWN_CLASSES_NAMES, nov_label=NOV_INDEX,
                                                        filepath=os.path.join(output_dir, 'results_frame.csv'))
        
        evaluation_dict = novelty_analysis.svm_evaluate_nov_detection(results_frame, filepath=os.path.join(output_dir, 'evaluation_dict.json'))
        metrics_keys = np.array(list(evaluation_dict.keys()))
        metrics_per_fold.append(list(evaluation_dict.values()))

        fold_end = time.time()
        elapsed_time = round((fold_end-fold_start), 2)
        folds_time[f'fold_{fold_count}']=elapsed_time

        print('-------------------------------------------------------------------------------')
        print(f'Elapsed time {elapsed_time} seconds')
        print('-------------------------------------------------------------------------------')

        output_dir, _ = os.path.split(output_dir)

        del svc_committee, predictions, results_frame, evaluation_dict
        gc.collect()        
        fold_count += 1

        if __name__ == '__main__':
            queue = collections.deque()
            t = threading.Thread(target=lambda x: x.append(input()), args=(queue,), daemon=True)
            print(f'If you want to pause press Enter {len(threading.enumerate())} times.\n')
            t.start()
            t.join(5.0)

            if len(queue) > 0:
                input('The script is paused press Enter to continue\n')
            
            del queue, t
            gc.collect()

    metrics_per_fold = np.array(metrics_per_fold)

    metrics_dict = {metric: {'folds': fold_metrics, 'avg': np.sum(fold_metrics)/len(fold_metrics), 'var': np.var(fold_metrics)}
                     for metric, fold_metrics in zip(metrics_keys, np.array(metrics_per_fold.T))}
    
    with open(os.path.join(output_dir, 'metrics_dict.json'), 'w') as json_file:
        json.dump(da_utils.cast_to_python(metrics_dict), json_file, indent=4)
    
    with open(os.path.join(output_dir, 'training_time.json'), 'w') as json_file:
        json.dump(da_utils.cast_to_python(folds_time), json_file, indent=4)

    del metrics_dict, folds_time
    gc.collect()
    if TELEGRAM_BOT:
        bot.sendMessage(CHAT_ID, MESSAGE_ID + 'Finished')

    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a+') as log:
        start = time.ctime() + f'\nTraining ended successfully\n'
        params = 'Params\n' + 'Model: SVM' + ' '.join([f'{key}: {value}' for key,value in args.__dict__.items()]) + '\n'
        log.write(start + params)

except Exception as e:
    error_message = f'{type(e)}: {e}'

    if TELEGRAM_BOT:
        bot.sendMessage(CHAT_ID, MESSAGE_ID + '\n' + error_message)
    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a+') as log:
        start = time.ctime() + f'\nTraining ended with an error\n{error_message}\n'
        params = 'Params\n' + 'Model: SVM' + ' '.join([f'{key}: {value}' for key,value in args.__dict__.items()]) + '\n'
        log.write(start + params)
    raise e