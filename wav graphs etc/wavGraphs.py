# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:50:19 2018

@author: WNP387
"""

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

width = 15
height = 3

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('bark of the pine tree.wav')
times = np.arange(len(data))/float(samplerate)

# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(width, height))
plt.title('wav file in time domain')

#plt.fill_between(times, data[:,0], data, color='b') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.plot(times,data)
plt.savefig('plot.jpeg', dpi=80)
plt.show()

# Plot the signal read from wav file
plt.figure(figsize=(width, height)) #first arg - length. second - height.
plt.title('Spectrogram of a wav file')
plt.specgram(data,Fs=samplerate) #left channel only
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

#spectre = np.fft.fft(data)
#freq = np.fft.fftfreq(data.size, 1/samplerate)

plt.figure(figsize=(width, height)) #first arg - length. second - height.
plt.title('phase of a wav file')
plt.phase_spectrum(data,Fs=samplerate)