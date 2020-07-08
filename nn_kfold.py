import os
import gc
import time
import json
import joblib
import socket
import getpass
import argparse
import telegram
import threading
import collections
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from setup import setup
from sklearn.metrics import recall_score, accuracy_score

#Gets enviroment variables and appends needed custom packages
LOFAR_FOLDER, OUTPUT_DIR, RAW_DATA = setup()

import my_models
import sonar_utils
from data_analysis.model_evaluation import novelty_analysis
from data_analysis.model_evaluation.splitters import RangedDataSplitter
from data_analysis.utils import math_utils
import data_analysis.utils.utils as da_utils
from data_analysis.utils import lofar_operators

parser = argparse.ArgumentParser(description='Trains a neural network model with sonar lofar data generated with the given parameters and cross-validates using kfold.',
                                    prog='Neual Network Kfold validation')
parser.add_argument('fft_pts', help='number of fft points used to generate the lofar data')
parser.add_argument('overlap', help='number of overlap used to generate the lofar data')
parser.add_argument('decimation', help='number of decimation used to generate the lofar data')
parser.add_argument('window_size', help='size of the window used to generate the sub images from the lofar data')
parser.add_argument('stride', help='stride that the window takes to generate the sub images from the lofar data')
parser.add_argument('novelty_class', help='classes that will be treated as novelty')
parser.add_argument('model', help='model instanciated in my_models to be used')
parser.add_argument('neurons', help='number of neurons in the intermediate layer')
parser.add_argument('folds', help='number of folds')
parser.add_argument('--bot', help='telegram bot token to report', default=None)
parser.add_argument('--chat', help='chat id for the bot', default=None)
args = parser.parse_args()

FFT_PTS = int(args.fft_pts)
OVERLAP = int(args.overlap)
DECIMATION = int(args.decimation)
WINDOW_SIZE = int(args.window_size)
STRIDE = int(args.stride)
NOVELTY_CLASS = args.novelty_class
MODEL_NAME = args.model
builder = getattr(my_models, MODEL_NAME)()
builder_class = type(builder)
FOLDS = int(args.folds)
NEURONS = int(args.neurons)
TELEGRAM_BOT = args.telegram
CHAT_ID = args.chat

#HERE THE SCRIPT REALLY STARTS
start_time = time.time()
folds_time = dict()

#CONSTANTS
#Contraints the maximum number of folds to 10
COLORS = ('#000000', '#008000', '#FF0000', '#FFFF00', '#0000FF', '#808080', '#FF00FF', '#FFA500', '#A52A2A', '#00FFFF')
if FOLDS>len(COLORS):
    raise ValueError(f'the maximum number of colors in this script for plotting is {len(COLORS)} please add more')

#Constants
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])       #Name of the classes
CLASSES_VALUES = np.arange(len(CLASSES_NAMES))      #Value of each class to be used in the model
CLASSES_DICT = {key:value for key, value in zip(CLASSES_NAMES, CLASSES_VALUES)}     #Dict conecting classes and its values

NOV_INDEX = list(CLASSES_NAMES).index(NOVELTY_CLASS)        #Index of the novelty class in CLASSES_NAMES
KNOWN_CLASSES_NAMES = np.delete(CLASSES_NAMES, NOV_INDEX)
KNOWN_CLASSES_VALUES = np.arange(len(KNOWN_CLASSES_NAMES))
KNOWN_CLASSES_DICT = {known_class: value 
            for known_class, value in zip(KNOWN_CLASSES_NAMES, KNOWN_CLASSES_VALUES)}
            #Dict with the key as known names and value as known values

#Constants functions
VALUE_MAPPING = lambda x: CLASSES_NAMES[CLASSES_VALUES == x][0]        #Maps the values back to the classes names
FROM_FULL_TO_KNOWN = lambda x: KNOWN_CLASSES_DICT[VALUE_MAPPING(x)]       #Function that connects full values to known values

THRESHOLD = novelty_analysis.create_threshold((200,450), ((-1,0.5),(0.500000001,1)))

#Getting a bot to give live updates on output
if TELEGRAM_BOT:
    MESSAGE_ID = f'{socket.gethostname()} at {getpass.getuser()} user.\n'
    bot = telegram.Bot(TELEGRAM_BOT)

try:        
    print(f'---------------------------- {FFT_PTS} FFT_PTS {DECIMATION} DECIMATION ----------------------------')

    data, labels, freq = joblib.load(os.path.join(LOFAR_FOLDER, f'lofar_data_file_fft_{FFT_PTS}_overlap_{OVERLAP}_decimation_{DECIMATION}_spectrum_left_400.npy'))
    runs_per_class = joblib.load(os.path.join(RAW_DATA, f'runs_info_{FFT_PTS}_fft_pts_{DECIMATION}_decimation.joblib'))
    runs_per_class = tuple(runs_per_class.values())
    #import pdb; pdb.set_trace()
    data = keras.utils.normalize(data)
    if builder_class.is_cnn:
        windows, win_labels, win_ranges = lofar_operators.window_runs(runs_per_class, np.arange(len(CLASSES_NAMES)), WINDOW_SIZE, STRIDE)
        splitter = RangedDataSplitter(windows, win_labels, win_ranges)
    elif builder_class.is_mlp:
        splitter = RangedDataSplitter(data, labels, runs_per_class)
    else:
        raise RuntimeError(f'The builder base architecture could not been identified. {builder_class} was passed')
    splitter.set_novelty(CLASSES_DICT[NOVELTY_CLASS], FROM_FULL_TO_KNOWN)
    

    output_dir = os.path.join(OUTPUT_DIR, f'lofar_parameters_{FFT_PTS}_fft_pts_{DECIMATION}_decimation', 
                                f'{MODEL_NAME}_intermediate_neurons{NEURONS}', 
                                f'kfold_novelty_class_{NOVELTY_CLASS}')

    fold_count = 1
    if builder_class.is_expert_committee:
        wrapper_eval_frames = list()
        exp_eval_frames = list()
    else:
        eval_frames = list()
    
    PARAMS = '\n'.join([f'{key}: {value}' for key,value in args.__dict__.items()])
    if TELEGRAM_BOT:
        message = MESSAGE_ID +'Starting the kfold split with the given parameters\n' + PARAMS
        bot.sendMessage(CHAT_ID, message)

    for x_test, y_test, x_val, y_val, x_train, y_train in splitter.kfold_split(n_splits=FOLDS, shuffle=True):
        fold_start = time.time()

        y_train = da_utils.to_sparse_tanh(y_train)
        y_val = da_utils.to_sparse_tanh(y_val)

        if builder_class.is_cnn:
            train_set = lofar_operators.LofarImgSequence(data, x_train, y_train)
            val_set = lofar_operators.LofarImgSequence(data, x_val, y_val)
            test_set = lofar_operators.LofarImgSequence(data, x_test, y_test)
        elif builder_class.is_mlp:
            train_set = da_utils.DataSequence(x_train, y_train)
            val_set = da_utils.DataSequence(x_val, y_val)
            test_set = lofar_operators.LofarImgSequence(x_test, y_test)

        #import pdb; pdb.set_trace()

        print('------------------------------ Control variables ------------------------------')
        print(f'current fold: {fold_count}')
        print(PARAMS) 
        print('-------------------------------------------------------------------------------')

        output_dir = os.path.join(output_dir, f'fold_{fold_count}')

        if TELEGRAM_BOT:
            bot.sendMessage(CHAT_ID, MESSAGE_ID + f'Starting the {MODEL_NAME} fitting for novelty_class {NOVELTY_CLASS} at fold {fold_count}')
        
        if builder_class.is_expert_committee:

            if TELEGRAM_BOT:
                bot.sendMessage(CHAT_ID, MESSAGE_ID + 'Fitting the expert_committee')
            committee_builder = builder.get_committee(x_test.input_shape(), KNOWN_CLASSES_NAMES, NEURONS, output_dir)
            committee_train_set = builder.change_to_expert_data(KNOWN_CLASSES_NAMES, KNOWN_CLASSES_DICT, train_set)
            committee_val_set = builder.change_to_expert_data(KNOWN_CLASSES_NAMES, KNOWN_CLASSES_DICT, val_set)

            if TELEGRAM_BOT:
                committee, committee_log = builder.fit_committee(committee_builder, committee_train_set, committee_val_set, bot, CHAT_ID)
                expert_message = MESSAGE_ID
                for class_, expert_log in committee_log.items():
                    expert_message += f'Expert for class {class_}\n' + sonar_utils.get_results_message(expert_log) + '\n'
                bot.sendMessage(CHAT_ID, expert_message)
                del expert_message
            else:
                committee, committee_log = builder.fit_committee(committee_builder, committee_train_set, committee_val_set)

            sonar_utils.save_expert_log(committee_log, output_dir)

            committee_predictions = committee.predict(test_set)
            
            exp_eval_frame = sonar_utils.eval_results(CLASSES_NAMES, THRESHOLD, NOVELTY_CLASS, NOV_INDEX, 
                                                        committee_predictions, y_test, output_dir, name_prefix='exp')
            exp_eval_frames.append(exp_eval_frame)

            del committee, committee_builder, committee_log, committee_train_set, committee_val_set

            if TELEGRAM_BOT:
                bot.sendMessage(CHAT_ID, MESSAGE_ID + 'Fitting the wrapper')

            train_wrapper = da_utils._WrapperSequence(committee.predict(train_set), train_set)
            val_wrapper = da_utils._WrapperSequence(committee.predict(val_set), val_set)
            wrapper = builder.get_wrapper(train_wrapper.input_shape())

            if TELEGRAM_BOT:
                wrapper_log = builder.fit_wrapper(wrapper, train_wrapper, val_wrapper, os.path.join(output_dir, 'wrapper'), bot, CHAT_ID)
                bot.sendMessage(CHAT_ID, MESSAGE_ID+'Committee wrapper results\n'+sonar_utils.get_results_message(wrapper_log))
            else:
                wrapper_log = builder.fit_wrapper(wrapper, train_wrapper, val_wrapper, os.path.join(output_dir, 'wrapper'))

            sonar_utils.save_multi_init_log(wrapper_log, output_dir)

            wrapper_predictions = wrapper.predict(x=committee_predictions)
            wrapper_eval_frame = sonar_utils.eval_results(CLASSES_NAMES, THRESHOLD, NOVELTY_CLASS, NOV_INDEX,
                                                            wrapper_predictions, y_test, output_dir, name_prefix='wrapper')
            eval_frames.append(wrapper_eval_frame)

            del train_wrapper, val_wrapper, wrapper

        else:
            model = builder.get_model(train_set.input_shape(), NEURONS)

            if TELEGRAM_BOT:
                log = builder.compile_and_fit(model, train_set, val_set, output_dir, bot, CHAT_ID)
                bot.sendMessage(CHAT_ID, MESSAGE_ID + sonar_utils.get_results_message(log))
            else:
                log = builder.compile_and_fit(model, train_set, val_set, output_dir)

            sonar_utils.save_multi_init_log(log, output_dir)

            predictions = model.predict(x=test_set)
            eval_frame = sonar_utils.eval_results(CLASSES_NAMES, THRESHOLD, NOVELTY_CLASS, NOV_INDEX,
                                                    predictions, y_test, output_dir)
            eval_frames.append(eval_frame)

            del model, log, predictions
        
        gc.collect()
        keras.backend.clear_session()

        np.save(os.path.join(output_dir, 'x_test.npy'), x_test)        
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)        
        np.save(os.path.join(output_dir, 'x_val.npy'), x_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)

        fold_end = time.time()
        elapsed_time = round((fold_end-fold_start), 2)
        folds_time[f'fold_{fold_count}']=elapsed_time

        print('-------------------------------------------------------------------------------')
        print(f'Elapsed time {elapsed_time} seconds')
        print('-------------------------------------------------------------------------------')

        #Pause routine
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
        
        fold_count += 1
        output_dir, _ = os.path.split(output_dir)

    if builder_class.is_expert_committee:
        exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict = sonar_utils.folds_eval(exp_eval_frames, 
                                                                                        output_dir, 
                                                                                        name_prefix='exp')
        sonar_utils.plot_data(NOVELTY_CLASS, FOLDS, COLORS, THRESHOLD,
                                exp_eval_frames, exp_eval_frames_avg, exp_eval_frames_var, exp_noc_area_dict,
                                output_dir, name_prefix='exp')

        wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_dict = sonar_utils.folds_eval(wrapper_eval_frames, 
                                                                                                    output_dir, 
                                                                                                    name_prefix='wrapper')
        sonar_utils.plot_data(NOVELTY_CLASS, FOLDS, COLORS, THRESHOLD,
                                wrapper_eval_frames, wrapper_eval_frames_avg, wrapper_eval_frames_var, wrapper_noc_dict,
                                output_dir, name_prefix='wrapper')

    else:
        eval_frames_avg, eval_frames_var, noc_area_dict = sonar_utils.folds_eval(eval_frames, output_dir)
        sonar_utils.plot_data(NOVELTY_CLASS, FOLDS, COLORS, THRESHOLD,
                                eval_frames, eval_frames_avg, eval_frames_var, noc_area_dict,
                                output_dir)

    with open(os.path.join(output_dir, 'parsed_params.json'), 'w') as json_file:
        json.dump(args.__dict__, json_file, indent=4)

    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a+') as log:
        start = time.ctime() + f'\nTraining finished successfully\n'
        log.write(start + PARAMS + '\n')

    if TELEGRAM_BOT:
        bot.sendMessage(CHAT_ID, MESSAGE_ID + 'Finished')

except Exception as e:
    error_message = f'{type(e)}: {e}'
    if TELEGRAM_BOT:
        bot.sendMessage(CHAT_ID, MESSAGE_ID + '\n' + error_message)
    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a+') as log:
        start = time.ctime() + f'\nTraining ended with an error\n{error_message}\n'
        log.write(start + PARAMS + '\n')
    raise e