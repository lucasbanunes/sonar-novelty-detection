import numpy as np

def window_run(run, window_size, stride):
    #import pdb; pdb.set_trace()
    run_start, run_end = run[0], run[-1]
    win_start = np.arange(run_start,run_end-window_size, stride)
    win_end = win_start + window_size
    #This removes window ends that may be out of range
    win_start = win_start[win_end <= run_end]
    win_end = win_end[win_end <= run_end]
    return np.apply_along_axis(lambda x: np.arange(*x), 1, np.column_stack((win_start, win_end)))

def windows_from_split(split, labels, window_size, stride):
    win_label = list()
    windows = list()
    #import pdb; pdb.set_trace()
    for run_split in split:
        window = window_run(run_split, window_size, stride)
        windows.append(window)
        win_label.append(np.full(len(window), labels[run_split][0]))
    
    win_label = np.concatenate(win_label)

    return windows, win_label
