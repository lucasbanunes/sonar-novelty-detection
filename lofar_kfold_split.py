import argparse
from config import setup
from utils.data_split import lofar_kfold_split

wav_folder, lofar_folder, _ = setup()

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

lofar_kfold_split(wav_folder, lofar_folder, fft_pts, overlap, decimation, bins, novelty_class, folds, window_size, stride, pca_components)