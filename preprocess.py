import numpy as np
import os
from scipy import io
import shutil
import pandas as pd

def preprocess(fpath, fname):

    # Create directory for pytorch data
    pytorch_path = os.path.join(fpath, fname[:-4] + '_pytorch')
    if os.path.isdir(pytorch_path):
        shutil.rmtree(pytorch_path)
    os.mkdir(pytorch_path)

    # Load dataset
    mat_file = io.loadmat(os.path.join(fpath, fname))
    data = mat_file['o'][0][0]['data']
    marker = np.array(mat_file['o'][0][0]['marker'], 
                           dtype='int8')
    sampFreq = int(mat_file['o'][0][0]['sampFreq'])

    # Inherent to experimental structure, trials are always 1 second
    trial_len = 1 * sampFreq

    # .mat file import format problems, probably there is a better 
    # way of handling this
    chnames = []
    for name in mat_file['o'][0][0]['chnames']:
        chnames.append(name[0][0])


    # It appears that the data sync channel is not present in every
    # recording. Also, data sync channel is named 'X5' but referred
    # to in Kaya et al as 'X3'?
    if 'X5' in chnames:
        data = np.delete(data, chnames.index('X5'), 
                              axis=1)
        chnames.remove('X5')

    # Digitally re-reference to common mode average
    ref = data.mean(axis=1)
    data = (data.transpose() - ref).transpose()

    # Remove ground ('A1', 'A2') channels 
    for ch in ['A1', 'A2']:
        loc = chnames.index(ch)
        data = np.delete(data, loc, axis=1)
        chnames.remove(ch)

    n_ch = len(chnames)

    # Find start indices of individual trials
    trial_start_inds = np.where(np.diff(marker, axis = 0) >= 1)[0] + 1
    # Set number of samples (trials)
    N = trial_start_inds.shape[0]
    # Allocate X and Y arrays
    X = np.zeros([n_ch, trial_len, N])
    Y = np.zeros(N, dtype=int)
    x_name = list()
    # Loop over trial start indices and chunk, save trial data
    for i in range(N):
        # Trial start index
        t_i = trial_start_inds[i]
        # Grab epoch / chunk
        X[0:, 0:, i] = np.swapaxes(data[t_i:t_i+trial_len, 0:], 0, 1)
        # Save X to file
        x_name.append(str(i) + '.npy')
        np.save(os.path.join(pytorch_path, x_name[-1]), X[0:, 0:, i])
        # Grab Y value
        Y[i] = int(marker[t_i])

    labels = pd.DataFrame(Y, x_name)
    annotations_file = os.path.join(pytorch_path, fname[:-4] + '.csv')
    # np.savetxt(annotations_file, Y, delimiter=",", fmt='%i')
    labels.to_csv(annotations_file)

if __name__ == "__main__":
    # Set data file path
    fpath = '/Volumes/SSD_DATA/kaya_mishchenko_eeg'
    fname = 'CLASubjectB1512153StLRHand.mat'
    preprocess(fpath, fname)
