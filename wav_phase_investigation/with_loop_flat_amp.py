
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment

#close all open figures
plt.close('all')

cartes = lambda r,theta: r * np.exp(1j*theta) #convert amp and phase to cartesian

#constants
height = 3; width = 15 #plot sizes   
axis = 0; rows = 1; columns = 1;
seg_size = 256
#splitting to 2 seperate wav files (by milliseconds).
t1 = 0; t2 = 6500; t3 = 13000 #Works in milliseconds


#split file to 2
wavFile = AudioSegment.from_wav("bark of the pine tree.wav")
newAudio = wavFile[t1:t2]# 6.5 first seconds
newAudio.export('part1_before.wav', format="wav") #Exports to a wav file in the current path.
newAudio2 = wavFile[t2:t3]
newAudio2.export('part2_before.wav', format="wav") #Exports to a wav file in the current path.

#read sample rate and wavData
samplerate, data1 = wavfile.read('part1_before.wav')
samplerate, data2 = wavfile.read('part2_before.wav')

data1 = data1[0:data1.shape[0] - np.mod(data1.shape[0],seg_size)] #so it will be divided evenly
data1 = np.split(data1,int(data1.shape[0]/seg_size)) #split to 256 sized arrays

data2 = data2[0:data2.shape[0] - np.mod(data2.shape[0],seg_size)] #so it will be divided evenly
data2 = np.split(data2,int(data2.shape[0]/seg_size)) #split to 256 sized arrays


#loop on every block of data
data1_i =[]
data2_i =[]

for data_block1, data_block2 in zip(data1,data2):
    #perform rfft on single block
    block_fft1 = np.fft.rfft(data_block1) #,axis = rows
    block_fft2 = np.fft.rfft(data_block2)

    #extract amp and phase (euler representation of complex number)
    r1 = np.absolute(block_fft1)
    theta1 = np.angle(block_fft1) #if you want angle in degrees, set (... , deg=True)
    
    r2 = np.absolute(block_fft2)
    theta2 = np.angle(block_fft2) #if you want angle in degrees, set (... , deg=True)
    
    r1[:] = 1.0; r2[:] = 1.0;

    #switch phases
    switched1 = cartes(r1,theta2) #part1 amplitude, with phase of part2  np.linspace(0,math.pi*2.0,52001)
    switched2 = cartes(r2,theta1) #part2 amplitude, with phase of part1

    #perform irfft on single block
    ifft1 = np.fft.irfft(switched1)#,axis = rows
    ifft2 = np.fft.irfft(switched2)
    
    absMaxIFFT1 = max(np.abs(ifft1)) 
    absMaxIFFT2 = max(np.abs(ifft2)) 

    normalizedIFFT1 = ifft1/absMaxIFFT1
    normalizedIFFT2 = ifft2/absMaxIFFT2
    
    #reconstruct data with switched phases
    data1_i.append(np.int16(normalizedIFFT1*2**15))
    data2_i.append(np.int16(normalizedIFFT2*2**15))


finalIFFT1 = (np.array(data1_i)).flatten()
finalIFFT2 = (np.array(data2_i)).flatten()

#write data back to file
wavfile.write("part1afterloop_flat_amp.wav",samplerate, finalIFFT1)
wavfile.write("part2afterloop_flat_amp.wav",samplerate, finalIFFT2)






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





    