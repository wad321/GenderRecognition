from glob import glob
from os import rename

iter = 0

for fname in glob('female_clips/*.wav'):
    newname = str(iter) + "_K.wav"
    rename(fname, newname)
    iter += 1

for fname in glob('male_clips/*.wav'):
    newname = str(iter) + "_M.wav"
    rename(fname, newname)
    iter += 1
