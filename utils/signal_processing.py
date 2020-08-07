from scipy.signal import decimate, convolve, spectrogram
from sklearn.preprocessing import normalize
import soundfile as sf
import numpy as np
import os

class SignalProcessor():

    def __init__(self, filepath):
        if os.path.isdir(filepath):
            self.filepath = filepath
        else:
            raise ValueError('The file path must be a directory with subdirectories for each class containg .wav files')

    def lofar_analysis(self, fft_pts, overlap, decimation, bins, norm=None):

        lofar_dataset = dict(class_labels = dict(), runs_per_class = dict(), data=list(), time=dict(), freq=None, labels=list(),
                                lofar_params=dict(fft_pts=fft_pts, overlap=overlap, decimation=decimation, bins=bins, norm=norm))

        range_start = 0
        class_label = 0

        for class_folder in np.sort(os.listdir(self.filepath)):
            current_dir = os.path.join(self.filepath, class_folder)
            if not os.path.isdir(current_dir):
                current_dir, _  = os.path.split(current_dir)
                continue

            new_class = True
            lofar_dataset['class_labels'][class_folder] = class_label

            for wav_file in np.sort(os.listdir(current_dir)):

                if not wav_file.split('.')[-1] == 'wav':
                    continue

                signal, sample_rate = sf.read(os.path.join(current_dir, wav_file))

                lofargram, freq, time = self.lofar(signal, sample_rate, fft_pts, overlap, decimation, bins)
                len_lofargram = len(lofargram)

                lofar_dataset['freq'] = freq
                if new_class:
                    lofar_dataset['runs_per_class'][class_folder] = [range(range_start, range_start+len_lofargram)]
                    lofar_dataset['time'][class_folder] = [time]
                    new_class = False
                else:
                    lofar_dataset['runs_per_class'][class_folder].append(range(range_start, range_start+len_lofargram))
                    lofar_dataset['time'][class_folder].append(time)

                lofar_dataset['data'].append(lofargram)
                lofar_dataset['labels'].append(np.full((len_lofargram,), class_label, np.int16))
                range_start += len_lofargram
            
            class_label += 1
            current_dir, _  = os.path.split(current_dir)

        lofar_dataset['data'] = np.concatenate(lofar_dataset['data'], axis=0)
        lofar_dataset['labels'] = np.concatenate(lofar_dataset['labels'], axis=0)

        if norm:
            normalize(lofar_dataset['data'], norm)

        return lofar_dataset

    def tpsw(self, x, npts=None, n=None, p=None, a=None):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if npts is None:
            npts = x.shape[0]
        if n is None:
            n=int(round(npts*.04/2.0+1))
        if p is None:
            p =int(round(n / 8.0 + 1))
        if a is None:
            a = 2.0
        if p>0:
            h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
        else:
            h = np.ones((1, 2*n+1))
            p = 1
        h /= np.linalg.norm(h, 1)

        apply_on_spectre = lambda spectre: convolve(h, spectre, mode='full')
        mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
        ix = int(np.floor((h.shape[0] + 1)/2.0)) # Defasagem do filtro
        mx = mx[ix-1:npts+ix-1] # Corrige da defasagem
        # Corrige os pontos extremos do espectro
        ixp = ix - p
        mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
        mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
        mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
        #return mx
        # Elimina picos para a segunda etapa da filtragem
        #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
        indl = (x-a*mx) > 0
        #x[indl] = mx[indl]
        x = np.where(indl, mx, x)
        mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
        mx=mx[ix-1:npts+ix-1,:]
        #Corrige pontos extremos do espectro
        mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
        mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais
        return mx

    def lofar(self, data, fs, n_pts_fft=1024, n_overlap=0,
        decimation_rate=3, spectrum_bins_left=None, **tpsw_args):
        #norm_parameters = {'lat_window_size': 10, 'lat_gap_size': 1, 'threshold': 1.3}
        
        if not isinstance(data, np.ndarray):
            raise ValueError('The data must be a numpy.ndarray')

        if decimation_rate > 1:
            dec_data = decimate(data, decimation_rate, 10, 'fir', zero_phase=True)
            dec_fs = fs/decimation_rate
        else:
            dec_data = data
            dec_fs = fs
        freq, time, power = spectrogram(dec_data,
                                        window=('hann'),
                                        nperseg=int(n_pts_fft),
                                        noverlap=n_overlap,
                                        nfft=n_pts_fft,
                                        fs=dec_fs,
                                        detrend=False,
                                        axis=0,
                                        scaling='spectrum',
                                        mode='magnitude')
        power = np.absolute(power)
        power = power / self.tpsw(power, **tpsw_args)
        power = np.log10(power)
        power[power < -0.2] = 0

        if spectrum_bins_left is None:
            spectrum_bins_left = power.shape[0]*0.8

        power = power[:spectrum_bins_left, :]
        freq = freq[:spectrum_bins_left]

        return np.transpose(power), freq, time