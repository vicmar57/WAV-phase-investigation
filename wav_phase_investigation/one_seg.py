from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
import math

axis = 0

#plot sizes    
width = 15
height = 3

seg_size = 256

#splitting to 2 seperate wav files (by milliseconds).
t1 = 0 #Works in milliseconds
t2 = 6500 #Works in milliseconds
t3 = 13000 #Works in milliseconds
wavFile = AudioSegment.from_wav("bark of the pine tree.wav")

#split file
newAudio = wavFile[t1:t2]# 6.5 first seconds
newAudio.export('part1_before.wav', format="wav") #Exports to a wav file in the current path.
newAudio2 = wavFile[t2:t3]
newAudio2.export('part2_before.wav', format="wav") #Exports to a wav file in the current path.

#read sample rate and wavData
#plot phase in t = [0,6500]ms
samplerate, data = wavfile.read('part1_before.wav')
#plt.figure(figsize=(width, height)) #plot with specific dimensions. first arg - length. second - height.
#plt.title('phase of bark of the pine tree.wav in t = [0,6500]ms')
#plt.phase_spectrum(data,Fs=samplerate)

#perform fft on data and split into r and theta 
#(euler representation of complex number)
fft1 = np.fft.rfft(data,axis =axis)
r1 = np.absolute(fft1)
theta1 = np.angle(fft1) #if you want angle in degrees, set (... , deg=True)

#theta1 = np.linspace(0,math.pi*2,104000)

#r1[:] =1.0

#plot phase in t = [6500,13000]ms
samplerate, data = wavfile.read('part2_before.wav')
#plt.figure(figsize=(width, height)) #first arg - length. second - height.
#plt.title('phase of bark of the pine tree.wav in t = [6500,13000]ms')
#plt.phase_spectrum(data,Fs=samplerate)

#perform fft on second part of data and split into r and theta 
#(euler representation of complex number)
fft2 = np.fft.rfft(data,axis =axis)
r2 = np.absolute(fft2)
theta2 = np.angle(fft2)

# element wise operation - convert to cartesian coordinates - back to "normal" representation.
cartes = lambda r,theta: r * np.exp(1j*theta)

#np.linspace(0,0,52001)

#switch phases between the two parts
switched1 = cartes(r1,theta2) #part1 amplitude, with phase of part2  np.linspace(0,math.pi*2.0,52001)
switched2 = cartes(r2,theta1) #part2 amplitude, with phase of part1

# inverse transform
ifft1 = np.fft.irfft(switched1,axis =axis)
ifft2 = np.fft.irfft(switched2,axis =axis)

#take only real part of ifft-s
realIFFT1 = np.real(ifft1)
realIFFT2 = np.real(ifft2)

absMaxIFFT1 = max(np.abs(realIFFT1)) 
absMaxIFFT2 = max(np.abs(realIFFT2)) 

normalizedIFFT1 = realIFFT1/absMaxIFFT1
normalizedIFFT2 = realIFFT2/absMaxIFFT2

finalIFFT1 = np.int16(realIFFT1)
finalIFFT2 = np.int16(realIFFT2)

#write data back to file
wavfile.write("part1after_oneSeg.wav",samplerate, finalIFFT1)
wavfile.write("part2after_oneSeg.wav",samplerate, finalIFFT2)

#show histogram of imaginary part of ifft1
#plt.figure(figsize=(width, height)) #first arg - length. second - height.
#plt.hist(np.imag(ifft1))


