import os
import gc
import time
import json
import socket
import getpass
import argparse
import telegram
import numpy as np
from tensorflow import keras

from config import setup
from utils.data_split import lofar_kfold_split
#Gets enviroment variables and appends needed custom packages
WAV_FOLDER, LOFAR_FOLDER, OUTPUT_DIR = setup()

import my_models
from data_analysis.neural_networks.training import MultiInitLog
from data_analysis.model_evaluation import novelty_analysis
from data_analysis.utils.utils import to_sparse_tanh

parser = argparse.ArgumentParser(description='Trains a neural network model with sonar lofar data generated with the given parameters and cross-validates using kfold.',
                                    prog='Neual Network Kfold validation')
parser.add_argument('model', help='model instanciated in my_models to be used')
parser.add_argument('neurons', help='number of neurons in the intermediate layer', type=int)
parser.add_argument('fft_pts', help='number of fft points used to generate the lofar data', type=int)
parser.add_argument('overlap', help='number of overlap used to generate the lofar data', type=int)
parser.add_argument('decimation', help='number of decimation used to generate the lofar data', type=int)
parser.add_argument('bins', help='number of bins retained from the lofargram', type=int)
parser.add_argument('novelty', help='class to be treated as novelty')
parser.add_argument('folds', help='number of folds', type=int)
parser.add_argument('--expert', '-e', help='expert to be wrapped', default=None)
parser.add_argument('--expert_neurons', '-en', help='number of intermediate neurons from the expert to be wrapped', default=None, type=int)
parser.add_argument('--window_size', '-w', help='size of the sliding window', default=None, type=int)
parser.add_argument('--stride', '-s', help='stride of the sliding window', default=None, type=int)
parser.add_argument('--pca', '-p', help='if passed defines the number of components to retain from a pca applied on the lofargram. -1 can be passed to retain all the components', 
                    default=None, type=int)
parser.add_argument('--bot', '-b', help='telegram bot token to report', default=None)
parser.add_argument('--chat', '-c', help='chat id for the bot', default=None)
parser.add_argument('--gpu', help='if true enables gpu training', default=False, type=bool)

args = parser.parse_args()
model_name = args.model
model_architecture = getattr(my_models, model_name)
neurons = args.neurons
fft_pts = args.fft_pts
overlap = args.overlap
decimation = args.decimation
bins = args.bins
novelty_class = args.novelty
folds = args.folds
if not args.expert is None:
    committee_achitecture = getattr(my_models, args.expert)
    committee_neurons = args.expert_neurons
window_size = args.window_size
stride = args.stride
pca_components = args.pca
if pca_components == -1:
    pca_components = bins
bot_token = args.bot
chat_id = args.chat
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#HERE THE SCRIPT REALLY STARTS
start_time = time.time()
folds_time = dict()

#Constants
CLASSES_NAMES = np.array(['A', 'B', 'C', 'D'])       #Name of the classes
CLASSES_VALUES = np.arange(len(CLASSES_NAMES))      #Value of each class to be used in the model

NOV_INDEX = list(CLASSES_NAMES).index(novelty_class)        #Index of the novelty class in CLASSES_NAMES
KNOWN_CLASSES_NAMES = np.delete(CLASSES_NAMES, NOV_INDEX)
KNOWN_CLASSES_VALUES = np.arange(len(KNOWN_CLASSES_NAMES))
KNOWN_CLASSES_DICT = {known_class: value 
            for known_class, value in zip(KNOWN_CLASSES_NAMES, KNOWN_CLASSES_VALUES)}
            #Dict with the key as known names and value as known values

EXPERT_MAPPING = {known_class: value 
            for known_class, value in zip(KNOWN_CLASSES_NAMES, to_sparse_tanh(KNOWN_CLASSES_VALUES))}

#Constants functions
VALUE_MAPPING = lambda x: CLASSES_NAMES[CLASSES_VALUES == x][0]        #Maps the values back to the classes names
TO_KNOWN_VALUES = lambda x: KNOWN_CLASSES_DICT[VALUE_MAPPING(x)]       #Function that connects the old known values to the new ones

THRESHOLD = novelty_analysis.create_threshold((200,450), ((-1,0.5),(0.500000001,1)))

datapath = os.path.join(LOFAR_FOLDER, 
                f'lofar_dataset_fft_pts_{fft_pts}_overlap_{overlap}_decimation_{decimation}_pca_{pca_components}_novelty_class_{novelty_class}_norm_l2')
if window_size is None:
    kfold_datapath = os.path.join(datapath, f'kfold_{folds}_folds', 'vectors')
else:
    kfold_datapath = os.path.join(datapath, f'kfold_{folds}_folds', f'window_size_{window_size}_stride_{stride}')

if not os.path.exists(kfold_datapath):
    lofar_kfold_split(WAV_FOLDER, LOFAR_FOLDER, fft_pts, overlap, decimation, bins, novelty_class, folds, window_size, stride, pca_components)

base_path = os.path.join(OUTPUT_DIR, f'lofar_parameters_{fft_pts}_fft_pts_{decimation}_decimation_pca_{pca_components}_bins_{bins}')
output_dir = os.path.join(base_path, f'{model_architecture.__name__}_intermediate_neurons_{neurons}', f'kfold_novelty_class_{novelty_class}')

#Getting a bot to give live updates on output
if bot_token:
    try:
        MESSAGE_ID = f'{socket.gethostname()} at {getpass.getuser()} user.\n'
        bot = telegram.Bot(bot_token)
    except telegram.error.TimedOut:
        pass
else:
    bot = None
    MESSAGE_ID = None

try:        
    print(f'---------------------------- {fft_pts} fft_pts {decimation} decimation ----------------------------')

    fold_count = 1
    
    PARAMS = '\n'.join([f'{key}: {value}' for key,value in args.__dict__.items()])
    if bot_token:
        try:
            message = MESSAGE_ID +'Starting the nn kfold training with the given parameters\n' + PARAMS
            bot.sendMessage(chat_id, message)
        except telegram.error.TimedOut:
            pass

    for split in range(folds):

        fold_start = time.time()
        
        print('------------------------------ Control variables ------------------------------')
        print(f'current fold: {fold_count}')
        print(PARAMS) 
        print('-------------------------------------------------------------------------------')

        output_dir = os.path.join(output_dir, f'fold_{fold_count}')

        if bot_token:
            try:
                bot.sendMessage(chat_id, MESSAGE_ID + f'Starting the {model_name} fitting for novelty_class {novelty_class} at fold {fold_count}')
            except telegram.error.TimedOut:
                pass
        
        if my_models.expert_commitee in model_architecture.__bases__:
            
            train_set, val_set, test_set = model_architecture.get_data(datapath, split, np.vectorize(TO_KNOWN_VALUES), args)
            experts = model_architecture.get_experts(train_set.input_shape(), neurons, KNOWN_CLASSES_NAMES)
            committee, experts_multi_inits = model_architecture.fit_committee(experts, 
                                                    KNOWN_CLASSES_NAMES, 
                                                    EXPERT_MAPPING, 
                                                    train_set, 
                                                    val_set,
                                                    bot,
                                                    chat_id,
                                                    MESSAGE_ID)

            for class_, multi_init in experts_multi_inits.items():
                save_dir = os.path.join(output_dir, 'experts_multi_init_log')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                multi_init.to_json(os.path.join(save_dir, f'{class_}_expert_multi_init_log.json'))
            
            committee_predictions = committee.predict(test_set)
            
            novelty_analysis.get_results(predictions=committee_predictions, labels=test_set.y_set, threshold=THRESHOLD, 
                                                            classes_names=CLASSES_NAMES, novelty_index=NOV_INDEX,
                                                            filepath=os.path.join(output_dir, 'results_frame.csv'))
        
        elif model_architecture is my_models.neural_committee:
            
            output_dir = os.path.join(base_path, 
                            f'{model_architecture.__name__}_intermediate_neurons_{neurons}_committee_{committee_achitecture.__name__}_neurons_{neurons}', 
                            f'kfold_novelty_class_{novelty_class}')
            
            multi_inits_experts = os.path.join(base_path, 
                                        f'{committee_achitecture.__name__}_intermediate_neurons_{committee_neurons}', 
                                        f'kfold_novelty_class_{novelty_class}', f'fold_{fold_count}', 'experts_multi_init_log')
            
            train_set, val_set, test_set = model_architecture.get_data(committee_achitecture, datapath, split, np.vectorize(TO_KNOWN_VALUES), args)
            
            wrapper_layers = model_architecture.get_wrapper_layers(neurons)
            neural_committee = model_architecture.get_model(multi_inits_experts, wrapper_layers, train_set.input_shape())
            multi_init_log = model_architecture.compile_and_fit(neural_committee, train_set, val_set, bot, chat_id, MESSAGE_ID)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            multi_init_log.to_json(os.path.join(output_dir, 'multi_init_log.json'))
            
            neural_committee.set_weights(multi_init_log.best_weights(metric='val_sparse_accuracy', mode='max', training_end=False))
            predictions = neural_committee.predict(x=test_set)

            novelty_analysis.get_results(predictions=predictions, labels=test_set.y_set, threshold=THRESHOLD, 
                                            classes_names=CLASSES_NAMES, novelty_index=NOV_INDEX,
                                            filepath=os.path.join(output_dir, 'results_frame.csv'))

            del multi_inits_experts, wrapper_layers, neural_committee, predictions, multi_init_log
            
        else:

            train_set, val_set, test_set = model_architecture.get_data(datapath, split, np.vectorize(TO_KNOWN_VALUES), args)
            model = model_architecture.get_model(train_set.input_shape(), neurons)
            multi_init = model_architecture.compile_and_fit(model, train_set, val_set, bot, chat_id, MESSAGE_ID)
            model.set_weights(multi_init.best_weights(metric='val_sparse_accuracy', mode='max', training_end=False))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            multi_init.to_json(os.path.join(output_dir, 'multi_init_log.json'))
            predictions = model.predict(x=test_set)
            novelty_analysis.get_results(predictions=predictions, labels=test_set.y_set, threshold=THRESHOLD, 
                                            classes_names=CLASSES_NAMES, novelty_index=NOV_INDEX,
                                            filepath=os.path.join(output_dir, 'results_frame.csv'))

            del model, predictions
        
        gc.collect()
        keras.backend.clear_session()
        fold_end = time.time()
        elapsed_time = round((fold_end-fold_start), 2)
        folds_time[f'fold_{fold_count}']=elapsed_time

        print('-------------------------------------------------------------------------------')
        print(f'Elapsed time {elapsed_time} seconds')
        print('-------------------------------------------------------------------------------')
        
        fold_count += 1
        output_dir, _ = os.path.split(output_dir)

    with open(os.path.join(output_dir, 'parsed_params.json'), 'w') as json_file:
        json.dump(args.__dict__, json_file, indent=4)

    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a') as log:
        start = time.ctime() + f'\nTraining finished successfully\n'
        log.write(start + PARAMS + '\n\n')

except Exception as e:
    error_message = f'{type(e)}: {e}'
    if bot_token:
        bot.sendMessage(chat_id, MESSAGE_ID + '\n' + error_message)
    with open(os.path.join(OUTPUT_DIR, 'training.log'), 'a') as log:
        start = time.ctime() + f'\nTraining ended with an error\n{error_message}\n'
        log.write(start + PARAMS + '\n\n')
    raise e