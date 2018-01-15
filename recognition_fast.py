import numpy as np
from sklearn.mixture import GMM
import python_speech_features as mfcc
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.externals import joblib
import glob
import sys
import os.path

import warnings

warnings.filterwarnings("ignore")

# NOT CHANGEABLE!
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


class GmmModel:
    m_train = male_training_size
    f_train = female_training_size
    comp = n_components
    iter = n_iter
    init = n_init
    winlen = winlen
    winstep = winstep
    numcep = numcep

    def save_variables(self):
        self.variable_list = [self.m_train, self.f_train, self.comp, self.iter,
                              self.init, self.winlen, self.winstep, self.numcep]

    def variable_check(self):
        if not hasattr(self, 'variable_list'):
            return False
        elif self.m_train == self.variable_list[0] and self.f_train == self.variable_list[1] \
                and self.comp == self.variable_list[2] and self.iter == self.variable_list[3] \
                and self.init == self.variable_list[4] and self.winlen == self.variable_list[5] \
                and self.winstep == self.variable_list[6] and self.numcep == self.variable_list[7]:
            return True
        else:
            return False


def gen_feature(rate, data):
    feature = mfcc.mfcc(data, rate / 2, winlen=winlen, winstep=winstep, numcep=numcep, appendEnergy=False, nfft=1024)
    return preprocessing.scale(feature)


def main():
    if os.path.isfile("models/male_model.pkl"):
        male_model = joblib.load("models/male_model.pkl")
    else:
        sys.exit("No fast male model found!")

    if os.path.isfile("models/female_model.pkl"):
        female_model = joblib.load("models/female_model.pkl")
    else:
        sys.exit("No fast female model found!")

    number_of_args = len(sys.argv)
    models = (male_model.gmm, female_model.gmm)
    names = ('M', 'K')

    if number_of_args > 1:
        for i in range(1, number_of_args):
            try:
                wav_rate, wav_data = wavfile.read(sys.argv[i])
                feature = gen_feature(wav_rate, wav_data)
                score = np.zeros(2)
                for j in range(2):
                    score[j] = np.array(models[j].score(feature)).sum()
                print(sys.argv[i], " : ", names[int(np.argmax(score))])
            except IOError:
                sys.exit(str("Cannot read argument number " + str(i) + "!"))


if __name__ == "__main__":
    main()
