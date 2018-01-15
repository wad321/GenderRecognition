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


def create_gmm(files):
    initialize_features = True
    features = []
    for file in files:
        wav_rate, wav_waves = wavfile.read(file)
        feature = gen_feature(wav_rate, wav_waves)
        if initialize_features:
            features = feature
            initialize_features = False
        else:
            features = np.append(features, feature, 0)
    gmm = GMM(n_components=n_components, n_iter=n_iter, n_init=n_init)
    gmm.fit(features)
    return gmm


def test_models(f_models, f_names, files):
    number_of_models = len(f_models)
    names_output = np.zeros(number_of_models)
    real_male_hits = 0
    real_female_hits = 0

    for file in files:
        wav_rate, wav_data = wavfile.read(file)
        feature = gen_feature(wav_rate, wav_data)
        score = np.zeros(number_of_models)
        for i in range(number_of_models):
            score[i] = np.array(f_models[i].score(feature)).sum()
        best_score = int(np.argmax(score))
        names_output[best_score] += 1
        print(file, " : ", f_names[best_score])
        if 'K' in file and 'K' in f_names[best_score]:
            real_female_hits += 1

        if 'K' not in file and 'K' not in f_names[best_score]:
            real_male_hits += 1

    print('Checked ', len(files), 'instances! Result is:')
    for i in range(number_of_models):
        print(f_names[i], " : ", int(names_output[i]))

    real_male_sample = 0
    real_female_sample = 0
    for i in files:
        if 'K' in i:
            real_female_sample += 1
        else:
            real_male_sample += 1

    if real_male_sample > 0:
        print('Actual values: M ', int(real_male_sample), " ; Correct M found: ", int(real_male_hits),
              ' ; Correct : ', (real_male_hits / real_male_sample) * 100, "%")

    if real_female_sample > 0:
        print('Actual values: K ', int(real_female_sample), " ; Correct F found: ", int(real_female_hits),
              ' ; Correct : ', (real_female_hits / real_female_sample) * 100, "%")

    if real_male_sample + real_female_sample > 0:
        print('Total correct: ', real_female_hits + real_male_hits, " out of ", real_female_sample + real_male_sample,
              ' ; Correct : ', (real_female_hits + real_male_hits) / (real_female_sample + real_male_sample) * 100, "%")

    return 0


def randomize_train_list(male_training, female_training):
    try:
        male_random_list = random.sample(range(0, len(glob.glob("male/*.wav"))), male_training)
        female_random_list = random.sample(range(0, len(glob.glob("female/*.wav"))), female_training)

        glob_m = glob.glob("male/*.wav")
        glob_f = glob.glob("female/*.wav")

        train_m = []
        for i in male_random_list:
            train_m.append(glob_m[i])

        train_f = []
        for i in female_random_list:
            train_f.append(glob_f[i])

        return train_m, train_f

    except ValueError:
        sys.exit("Sample size exceeded population size.")


def main():
    train_m, train_f = randomize_train_list(male_training_size, female_training_size)

    if os.path.isfile("models/male_model.pkl"):
        male_model = joblib.load("models/male_model.pkl")
        if not male_model.variable_check():
            male_model = GmmModel()
            male_model.gmm = create_gmm(train_m)
    else:
        male_model = GmmModel()
        male_model.gmm = create_gmm(train_m)

    if os.path.isfile("models/female_model.pkl"):
        female_model = joblib.load("models/female_model.pkl")
        if not female_model.variable_check():
            female_model = GmmModel()
            female_model.gmm = create_gmm(train_f)
    else:
        female_model = GmmModel()
        female_model.gmm = create_gmm(train_f)
        
    male_model.save_variables()
    female_model.save_variables()
    joblib.dump(male_model, "models/male_model.pkl")
    joblib.dump(female_model, "models/female_model.pkl")

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
                
    else:
    # Method to predict from folder path (default is 'test/*.wav')
        if len(glob.glob("test/*.wav")) > 0:
            test_models(models, names, glob.glob("test/*.wav"))  # Calculates overall accuracy, accepts .wav files,
    # female file MUST have a letter 'K' in file name, male must NOT have a 'K' in file name

    
if __name__ == "__main__":
    main()
