import numpy as np

def window_run(run_, window_size, stride):
    if len(run_) == window_size:
        return np.array([run_])
    run_start, run_end = run_[0], run_[-1]
    win_start = np.arange(run_start,run_end-window_size+1, stride)
    win_end = win_start + window_size
    #This removes window ends that may be out of range
    win_start = win_start[win_end <= run_end]
    win_end = win_end[win_end <= run_end]
    return np.apply_along_axis(lambda x: np.arange(*x), 1, np.column_stack((win_start, win_end)))

def windows_from_split(split, labels, window_size, stride):
    win_label = list()
    windows = list()
    for run_split in split:
        if len(run_split) < window_size:
            continue
        window = window_run(run_split, window_size, stride)
        windows.append(window)
        win_label.append(np.full(len(window), labels[run_split][0]))
    
    win_label = np.concatenate(win_label)
    windows = np.concatenate(windows, axis=0)

    return windows, win_label