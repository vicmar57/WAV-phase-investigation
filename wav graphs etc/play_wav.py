# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:31:10 2018

@author: wnp387
"""

#worksssssss
import pyaudio 
import wave  
 
#define stream chunk   
chunk = 1024  
 
#open a wav format music  
f = wave.open("futurebells_minbass.wav","rb")  
#instantiate PyAudio  
p = pyaudio.PyAudio()  
#open stream  
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
#read data  
data = f.readframes(chunk)  
 
#play stream  
while data:  
    stream.write(data)  
    data = f.readframes(chunk)  
 
#stop stream  
stream.stop_stream()  
stream.close()  
 
#close PyAudio  
p.terminate() 