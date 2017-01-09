#!/usr/bin/env python
from python_speech_features import *
from python_speech_features import mfcc, logfbank, ssc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
timit_path = '/home/c2tao/timit_mini_corpus/'
files = map(lambda x: os.path.join(timit_path, x), os.listdir(timit_path))

(rate,sig) = wav.read(files[0])
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate, nfilt=32)
ssc_feat = ssc(sig,rate)

print fbank_feat.shape
print mfcc_feat.shape
print ssc_feat.shape

plt.matshow(fbank_feat)
plt.show()
