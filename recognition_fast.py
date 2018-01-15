import numpy as np
from sklearn.mixture import GMM
import python_speech_features as mfcc
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.externals import joblib
import glob
import random
import sys
import os.path

import warnings

warnings.filterwarnings("ignore")

# Variables to change
# Please be advised, changing arguments will result in preparing another model, which might be time consuming.
male_training_size = len(glob.glob('male/*.wav'))  # Size of male training model,
# max - (number of .wav files in 'male/')
female_training_size = len(glob.glob('female/*.wav'))  # Size of female training model,
# max - (number of .wav files in 'female/')
n_components = 10  # Default: 1
n_iter = 200  # Default: 100
n_init = 5  # Default: 1
winlen = 0.025  # Default: 0.025
winstep = 0.01  # Default: 0.01
numcep = 13  # Default: 13
# End of variables to change


def gen_feature(rate, data):
    feature = mfcc.mfcc(data, rate / 2, winlen=winlen, winstep=winstep, numcep=numcep, appendEnergy=False, nfft=1024)
    return preprocessing.scale(feature)


def main():
    if os.path.isfile("models/male_model.pkl"):
        male_model = joblib.load("models/male_model.pkl")
        if not male_model.variable_check():
            sys.exit("No fast male model found!")
    else:
        sys.exit("No fast male model found!")

    if os.path.isfile("models/female_model.pkl"):
        female_model = joblib.load("models/female_model.pkl")
        if not female_model.variable_check():
            sys.exit("No fast female model found!")
    else:
        sys.exit("No fast female model found!")

    number_of_args = len(sys.argv)
    models = (male_model.gmm, female_model.gmm)
    names = ('M', 'K')

    if number_of_args > 1:
        for i in range(1, number_of_args):
            try:
                feature = gen_feature(wavfile.read(sys.argv[i]))
                score = np.zeros(2)
                for j in range(2):
                    score[j] = np.array(models[j].score(feature)).sum()
                print(sys.argv[i], " : ", names[int(np.argmax(score))])
            except IOError:
                sys.exit(str("Cannot read argument number " + str(i) + "!"))


if __name__ == "__main__":
    main()