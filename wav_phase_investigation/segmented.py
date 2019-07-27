from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import math
from pydub import AudioSegment


#plot sizes    
width = 15
height = 3

axis = 0
rows = 0
columns = 1

seg_size = 256

#splitting to 2 seperate wav files (by milliseconds).
t1 = 0 #Works in milliseconds
t2 = 6500 #Works in milliseconds
t3 = 13000 #Works in milliseconds

plt.close('all')

#split file
wavFile = AudioSegment.from_wav("bark of the pine tree.wav")
newAudio = wavFile[t1:t2]# 6.5 first seconds
newAudio.export('part1before.wav', format="wav") #Exports to a wav file in the current path.
newAudio2 = wavFile[t2:t3]
newAudio2.export('part2before.wav', format="wav") #Exports to a wav file in the current path.

#read sample rate and wavData
samplerate, data = wavfile.read('part1before.wav')

data = data[0:data.shape[0] - np.mod(data.shape[0],seg_size)] #so it will be divided evenly
#data =   np.split(data,int(data.shape[0]/seg_size)) #split to 256 sized arrays
data = np.reshape(data,[-1,256]) #split to 256 sized arrays

#perform fft on data and split into r and theta 
#(euler representation of complex number)
#fftn1 = [np.fft.fft(block) for block in data]
fftn1 = np.fft.fftn(data,axes =[axis])
#r1 = np.ones(fftn1.shape)
r1 = np.absolute(fftn1)
theta1 = np.angle(fftn1) #if you want angle in degrees, set (... , deg=True)


samplerate, data = wavfile.read('part2before.wav')
#perform fft on second part of data and split into r and theta 
#(euler representation of complex number)
data = data[0:data.shape[0] - np.mod(data.shape[0],seg_size)] #so it will be divided evenly
data = np.reshape(data,[-1,256]) #split to 256 sized arrays

#fftn2 = [np.fft.fft(block) for block in data]
fftn2 = np.fft.fftn(data,axes =[axis])
r2 = np.absolute(fftn2)
theta2 = np.angle(fftn2)

# element wise operation - convert to cartesian coordinates - back to "normal" representation.
cartes = lambda r,theta: r * np.exp(1j*theta)

#build conjugate symmetric phase
#conjSymPhase = np.random.rand(406,256)*math.pi
#conjSymPhase [:,128:] = np.flip(conjSymPhase[:,:128],1)*(-1)

 
#conjSymPhase = np.zeros((406,seg_size))
#r = np.concatenate([np.linspace(0,math.pi*2,seg_size/2),np.linspace(-math.pi*2,0,seg_size/2)])
#
#for row in range(conjSymPhase.shape[0]):
#    conjSymPhase[row][:] = r
#conjSymPhase = [np.linspace(-math.pi,math.pi,256) for i in range(conjSymPhase.shape[0])]
#conjSymPhase [:,128:] = np.flip(conjSymPhase[:,:128],1)*(-1)
#
#plt.figure(figsize=(width, height)) #first arg - length. second - height.
#plt.grid(True)
#plt.title('phase of bark of the pine tree.wav in t = [6500,13000]ms')
#plt.plot(conjSymPhase[17][:])


#switch phases between the two parts
switched1 = cartes(r1,theta2)  # np.random.rand(406,256)*2*math.pi   conjSymPhase
switched2 = cartes(r2,theta1)

# inverse transform
ifft1 = np.fft.ifftn(switched1,axes =[axis])
ifft2 = np.fft.ifftn(switched2,axes =[axis])

#take only real part of ifft-s
realIFFT1 = np.real(ifft1)
realIFFT2 = np.real(ifft2)

absMaxIFFT1 = np.amax(np.abs(realIFFT1)) 
absMaxIFFT2 = np.amax(np.abs(realIFFT2)) 

normalizedIFFT1 = realIFFT1/absMaxIFFT1
normalizedIFFT2 = realIFFT2/absMaxIFFT2

finalIFFT1 = np.int16(realIFFT1)
finalIFFT2 = np.int16(realIFFT2)

finalIFFT1 = finalIFFT1.flatten()
finalIFFT2 = finalIFFT2.flatten()

#im1 = np.reshape(np.imag(ifft1),[-1,1])
#print(min(im1))
#print(max(im1))
plt.figure(figsize=(width, height)) #first arg - length. second - height.
plt.title('amp of bark of the pine tree.wav in t = [6500,13000]ms')
plt.grid(True)
plt.plot(r1[2,:])

#write data back to file
wavfile.write("part1after.wav",samplerate, finalIFFT1)
wavfile.write("part2after.wav",samplerate, finalIFFT2)

#plt.figure(figsize=(width, height)) #first arg - length. second - height.
#plt.hist(np.imag(ifft1))
#samplerate, data = wavfile.read('part1before.wav')
#data = data[0:data.shape[0] - np.mod(data.shape[0],seg_size)] #so it will be divided evenly
#data =   np.split(data,int(data.shape[0]/seg_size)) #split to 256 sized arrays
#data1_i =[]
#data2_i =[]
#for block1, block2 in zip(data,data):
#    data1_i.append(block1 )
#    data2_i.append(block2 )
    