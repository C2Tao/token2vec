from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os
timit_path = '/home/c2tao/timit_mini_corpus/'
files = map(lambda x: os.path.join(timit_path, x), os.listdir(timit_path))
print files


samplerate, samples = wavfile.read(files[1])

from pylab import *
specgram(samples)

plt.savefig('foo.png')
'''
f, t, Sxx = spectrogram(samples, fs = samplerate)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''
