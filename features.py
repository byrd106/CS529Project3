import os
import scikits.audiolab
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.audiolab import Sndfile
from scikits.audiolab import play
from scikits.talkbox.features import mfcc

def readWAVFile(path):
	sr,x = scipy.io.wavfile.read(path)
	return x

def readAUFile(path):
	f = Sndfile(path, 'r')
	data = f.read_frames(f.nframes)
	return data

def allFeatures(data):
	f1 = getF1(data)
	f2 = getF2(data)
	featureSet = np.append(f1,f2)
	return featureSet

def getF3(data):
	ceps, mspec, spec = lpcc(data)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

def getF2(data):
	ceps, mspec, spec = mfcc(data)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

def getF1(data):
	#f = Sndfile('/Users/danielbyrd/Desktop/AppDevelopment/class/ML3/genres/blues/blues.00000.au', 'r')
	#f = Sndfile('/Users/danielbyrd/Desktop/AppDevelopment/class/ML3/classical.00000.au', 'r')
	#sr,x = scipy.io.wavfile.read('/Users/danielbyrd/Desktop/AppDevelopment/class/ML3/classical.00000.wav')
	#data = data.read_frames(661794)
	data = abs(scipy.fft(data)[:1000])
	return data
	#here is feature one 
	#print data.shape

	# data = f.read_frames(1000)
	# print data[:10],x[:10]
	# print sr
	# print f.samplerate
	# print f.encoding

	#print scipy.fft(x)[:1000].shape
	#fft_f = abs(scipy.fft(x)[:1000])

	# fs = f.samplerate
	# nc = f.channels
	# enc = f.encoding
	# data = f.read_frames(661794)
	# 30 second file 
	# data = f.read_frames(661794)
	# data = scipy.fft(data)
	# data = abs(data[:1000])
	# print fft_f[0],fft_f[100],fft_f[200]
	# print data[0],data[100],data[200]