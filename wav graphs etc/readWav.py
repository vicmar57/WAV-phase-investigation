from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

#plot sizes    
width = 15
height = 3

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('bark of the pine tree.wav')

#splitting to 2 seperate wav files (by milliseconds).
from pydub import AudioSegment
t1 = 0 #Works in milliseconds
t2 = 6500 #Works in milliseconds
t3 = 13000 #Works in milliseconds

#read wav file into wavFile
wavFile = AudioSegment.from_wav("bark of the pine tree.wav")

#split file
newAudio = wavFile[t1:t2]# 6.5 first seconds
newAudio.export('part1before.wav', format="wav") #Exports to a wav file in the current path.
newAudio2 = wavFile[t2:t3]
newAudio2.export('part2before.wav', format="wav") #Exports to a wav file in the current path.

#read sample rate and wavData
#plot phase in t = [0,6500]ms
samplerate, data = wavfile.read('part1before.wav')
plt.figure(figsize=(width, height)) #plot with specific dimensions. first arg - length. second - height.
plt.title('phase of bark of the pine tree.wav in t = [0,6500]ms')
plt.phase_spectrum(data,Fs=samplerate)

#perform fft on data and split into r and theta 
#(euler representation of complex number)
fft1 = np.fft.fft(data)
r1 = np.absolute(fft1)
r1[:] =1.0
theta1 =  np.angle(fft1) #if you want angle in degrees, set (... , deg=True)

#plot phase in t = [6500,13000]ms
samplerate, data = wavfile.read('part2before.wav')
plt.figure(figsize=(width, height)) #first arg - length. second - height.
plt.title('phase of bark of the pine tree.wav in t = [6500,13000]ms')
plt.phase_spectrum(data,Fs=samplerate)

#perform fft on second part of data and split into r and theta 
#(euler representation of complex number)
fft2 = np.fft.fft(data)
r2 = np.absolute(fft2)
theta2 =  np.angle(fft2)

# element wise operation - convert to cartesian coordinates - back to "normal" representation.
cartes = lambda r,theta: r * np.exp(1j*theta)

#switch phases between the two parts
switched1 = cartes(r1,theta2)
switched2 = cartes(r2,theta1)

# inverse transform
ifft1 = np.fft.ifft(switched1)
ifft2 = np.fft.ifft(switched2)

#take only real part of ifft-s
realIFFT1 = np.real(ifft1)
realIFFT2 = np.real(ifft2)

absMaxIFFT1 = max(np.abs(realIFFT1)) 
absMaxIFFT2 = max(np.abs(realIFFT2)) 

normalizedIFFT1 = realIFFT1/absMaxIFFT1
normalizedIFFT2 = realIFFT2/absMaxIFFT2

finalIFFT1 = np.int16(normalizedIFFT1 *2**15)
finalIFFT2 = np.int16(normalizedIFFT2 *2**15)

#write data back to file
wavfile.write("part1after.wav",samplerate, finalIFFT1)
wavfile.write("part2after.wav",samplerate, finalIFFT2)


plt.figure(figsize=(width, height)) #first arg - length. second - height.
plt.hist(np.imag(ifft1))














# =============================================================================
# import matplotlib.pyplot as plt
# from scipy.io import wavfile as wav
# from scipy.fftpack import fft
# import numpy as np
# rate, data = wav.read('2kHz_tone.wav')
# fft_out = fft(data)
# # matplotlib inline
# plt.figure(figsize=(20, 4)) #first arg - length. second - height.
# plt.plot(np.angle(fft_out))
# plt.show()



#phase = phase(data[:,0])
#mask=freq>0   
#mask = mask(0:len(mask)/2)
#mask = np.resize(mask,(len(data),1))
#var1 = freq[mask]
#var2 = np.abs(spectre[mask])
#plt.plot(freq[mask],np.abs(spectre[mask]))
#plt.figure(figsize=(20, 4)) #first arg - length. second - height.
#plt.title('phase of a wav file')
#plt.plot(freq,np.angle(spectre))
#plt.xlabel('Time')
#plt.ylabel('Frequency')
#plt.show()
# =============================================================================
