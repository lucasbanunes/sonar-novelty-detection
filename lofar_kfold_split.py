import argparse
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA

from config import setup
from utils.signal_processing import SignalProcessor
from utils.lofar_operators import windows_from_split

wav_folder, lofar_folder, _ = setup()

from data_analysis.model_evaluation.splitters import most_even_split

parser = argparse.ArgumentParser()
parser.add_argument('fft_pts', help='number of fft points used to generate the lofar data', type=int)
parser.add_argument('overlap', help='number of overlap used to generate the lofar data', type=int)
parser.add_argument('decimation', help='number of decimation used to generate the lofar data', type=int)
parser.add_argument('bins', type=int)
parser.add_argument('novelty', help='class to be treated as novelty during the pca to test data reconstruction')
parser.add_argument('folds', help='number of folds', type=int)
parser.add_argument('--window_size', '-w', help='size of the sliding window', default=None, type=int)
parser.add_argument('--stride', '-s', help='stride of the sliding window', default=None, type=int)
parser.add_argument('--pca', '-p', help='if passed defines the number of components to consider with a pca applied on the lofargram', default=None, type=int)

args = parser.parse_args()
fft_pts = args.fft_pts
overlap = args.overlap
decimation = args.decimation
bins = args.bins
novelty_class = args.novelty
folds = args.folds
window_size = args.window_size
stride = args.stride
pca_components = args.pca
if pca_components == -1:
    pca_components = bins

generate(fft_pts, overlap, decimation, bins, novelty_class, folds, window_size, stride, pca_components)

def generate(fft_pts, overlap, decimation, bins, novelty_class, folds, window_size, stride, pca_components):

    base_path = os.path.join(lofar_folder, 
                    f'lofar_dataset_fft_pts_{fft_pts}_overlap_{overlap}_decimation_{decimation}_pca_{pca_components}_novelty_class_{novelty_class}_norm_l2')
    lofar_path = os.path.join(base_path, 'data.z')
    cache_data = os.path.exists(lofar_path)

    if cache_data:
        print('Given lofar dataset already exists')
        lofar_dataset = joblib.load(lofar_path)
    else:
        print('Lofar dataset does not exist, creating one.')
        os.makedirs(base_path)
        sonar_dataset = SignalProcessor(wav_folder)
        lofar_dataset = sonar_dataset.lofar_analysis(fft_pts, overlap, decimation, bins, 'l2')

    known_runs_per_class = lofar_dataset['runs_per_class'].copy()
    novelty_data = (novelty_class, known_runs_per_class.pop(novelty_class))
    known_runs_range = np.concatenate(tuple(known_runs_per_class.values()))

    if not cache_data:

        if not pca_components is None:
            
            print('Applying PCA')
            pca = PCA(pca_components)
            pca.fit(lofar_dataset['data'][np.hstack(known_runs_range)])
            lofar_dataset['data'] = pca.transform(lofar_dataset['data'])
        
        joblib.dump(lofar_dataset, lofar_path)

    splits = np.array([most_even_split(run, folds) for run in known_runs_range]).T

    for test_split in range(folds):

        print(f'Mouting split {test_split}')

        if test_split == folds-1:
            val_split = 0
        else:
            val_split = test_split + 1
        splits_index = np.arange(len(splits))
        train_splits = np.all(np.stack((splits_index != test_split,  splits_index != val_split),axis=0), axis=0)
        to_save = dict(class_labels=np.array(list(lofar_dataset['class_labels'].items())))

        if not window_size is None:
            to_save['x_test'], to_save['y_test'] = windows_from_split(splits[test_split], lofar_dataset['labels'], window_size, stride)
            to_save['x_val'], to_save['y_val'] = windows_from_split(splits[val_split], lofar_dataset['labels'], window_size, stride)
            to_save['x_train'], to_save['y_train'] = windows_from_split(np.hstack(splits[train_splits]), lofar_dataset['labels'], window_size, stride)
            to_save['x_novelty'], to_save['y_novelty'] = windows_from_split(novelty_data[-1], lofar_dataset['labels'], window_size, stride)
            to_save['data'] = lofar_dataset['data']

            save_folder = os.path.join(base_path, f'kfold_{folds}_folds', f'window_size_{window_size}_stride_{stride}')
        else:      
            test_index = np.hstack(splits[test_split])
            val_index = np.hstack(splits[val_split])
            train_index = np.hstack(np.hstack(splits[train_splits]))
            novelty_index = np.hstack(novelty_data[-1])

            to_save['x_test'] = lofar_dataset['data'][test_index]
            to_save['x_val'] = lofar_dataset['data'][val_index]
            to_save['x_train'] = lofar_dataset['data'][train_index]
            to_save['x_novelty'] = lofar_dataset['data'][novelty_index]
            to_save['y_test'] = lofar_dataset['labels'][test_index]
            to_save['y_val'] = lofar_dataset['labels'][val_index]
            to_save['y_train'] = lofar_dataset['labels'][train_index]
            to_save['y_novelty'] = lofar_dataset['labels'][novelty_index]

            save_folder = os.path.join(base_path, f'kfold_{folds}_folds', 'vectors')

        print('Saving')

        if not os .path.exists(save_folder):
            os.makedirs(save_folder)
        
        np.savez(os.path.join(save_folder, f'split_{test_split}'), **to_save)