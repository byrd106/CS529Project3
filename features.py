import math
import os
import scikits.audiolab
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.audiolab import Sndfile
from scikits.audiolab import play
#from scikits.talkbox.features import mfcc
import librosa
from fileOperations import *
import random
from sklearn import decomposition
from scipy import signal

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def crossZero(data):
	zcr = librosa.feature.zero_crossing_rate(data)[0][:1100]
	return zcr

def smoothCrossZero(data):
	win = signal.hann(50)
	zcr = librosa.feature.zero_crossing_rate(data)[0][:1100]
	sconvoler = signal.convolve(zcr, win, mode='same') / sum(win)
	return sconvoler

def omfcc(data):
	#mfcc_feat = mfcc(data,22050)
	fbank_feat = logfbank(data,22050)
	return fbank_feat[1:2,:][0]

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
	f3 = getF3(data)
	featureSet = np.append(f1,f2)
	featureSet = np.append(featureSet,f3)
	return featureSet

def twoFeatures(data):
	f2 = getF2(data)
	f3 = getF3(data)
	featureSet = np.append(f2,f3)
	return featureSet


def getF4(data):
	return [2,random.choice([1,2,3,4])]

def getF5(data):
	return [2,random.choice([1,2,3,4])]

def flatChroma(data):
	#sr = 22050
	# b_tempo, b_beat_frames = librosa.beat.beat_track(y=blues, sr=sr)
	# b_harmonic, y_percussive = librosa.effects.hpss(blues)
	# b_chromagram = librosa.feature.chroma_cqt(y=b_harmonic,sr=sr)
	# b_beat_chroma = librosa.util.sync(b_chromagram,b_beat_frames,aggregate=np.median)

	sr = 22050
	lset = librosa.feature.chroma_stft(y=data, sr=sr)

	a = []
	for g in lset:
		a.append(np.mean(g))
		a.append(np.max(g))
		a.append(np.min(g))
		a.append(np.std(g))
		a.append(np.var(g))

	return a

def spectralCentroidStats(data):
	lz = librosa.feature.spectral_centroid(data)[0]
	return [np.max(lz),np.std(lz),np.mean(lz),np.min(lz),np.var(lz)]

def centroid(data):
	return librosa.feature.spectral_centroid(data)[0][:1000]

def waveStats(k):
	summaryStats = [np.max(k),np.std(k),np.mean(k),np.min(k),np.var(k)]
	return summaryStats

def zeroCrossingMFCC(data):
	k = librosa.feature.zero_crossing_rate(data)[0]
	ceps, mspec, spec = mfcc(k)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

def harmonicPercussiveSumStats(data):
	y_harmonic, y_percussive = librosa.effects.hpss(data)
	summaryStats = [ 
			np.max(y_harmonic),np.std(y_harmonic),np.mean(y_harmonic),np.min(y_harmonic),np.var(y_harmonic),
			np.max(y_percussive),np.std(y_percussive),np.mean(y_percussive),np.min(y_percussive),np.var(y_percussive) 
	]
	return summaryStats

def basicMFCC(k):
	ceps, mspec, spec = mfcc(k)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

def harmonicPercussiveMFCC(data):
	harmonic, percussive = librosa.effects.hpss(data)
	a = basicMFCC(harmonic).tolist()
	b = basicMFCC(percussive).tolist()
	return a+b

def spectralRoll(data):
	return librosa.feature.spectral_rolloff(data)[0][:1000]

def spectralRollStats(data):
	sk = librosa.feature.spectral_rolloff(data)[0]
	return [np.max(sk),np.std(sk),np.mean(sk),np.min(sk),np.var(sk)]

def onsetStats(data):
	sr = 22050
	onset_env = librosa.onset.onset_strength(y=data, sr=sr,aggregate=np.median, fmax=8000, n_mels=256)
	return [np.mean(onset_env),np.var(onset_env),np.std(onset_env),np.max(onset_env),np.min(onset_env)]

def tempo(data):
	sr = 22050
	y_harmonic, y_percussive = librosa.effects.hpss(data)
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
	                                             sr=sr)
	return tempo

def chromaData(data):
	sr = 22050
	y_harmonic, y_percussive = librosa.effects.hpss(data)
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
	                                             sr=sr)
	chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
	                                        sr=sr)
	beat_chroma = librosa.util.sync(chromagram,
	                                beat_frames,
	                                aggregate=np.median)
	beat_features = np.vstack([beat_chroma])
	return beat_features.flatten()[:300]

def RMSEStats(data):
	S = librosa.magphase(librosa.stft(data, window=np.ones, center=False))[0]
	S = librosa.feature.rmse(S=S)
	return [np.mean(S),np.var(S),np.std(S),np.max(S),np.min(S)]

def RMSE(data):
	S = librosa.magphase(librosa.stft(data, window=np.ones, center=False))[0]
	dout = librosa.feature.rmse(S=S)[0][:1000]
	return dout

def getF3(data):
	sr = 22050 # default sampling rate for our music
	k = librosa.feature.zero_crossing_rate(data)[0]
	summaryStats = [np.max(k),np.std(k),np.mean(k),np.min(k),np.var(k)]
	return summaryStats

def getF2(data):
	ceps, mspec, spec = mfcc(data)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

def getF1(data):
	data = abs(scipy.fft(data)[:1000])
	return data
	
def createFeature(featureFunction,name):
	FILE_PATH = "genres/"
	outputDataset = []
	dataset = getTotalDataset()
	count = 0
	for key in dataset.keys():
		for row in dataset[key]:
			data = [666,667]
			if count not in data:
				row = np.append(featureFunction(readAUFile(FILE_PATH+key+"/"+row)),mapKeyToInt(key))
				if math.isnan(float(row[1])):
					print count
				else:
					outputDataset.append(row)
			count+=1
	listToCSV(outputDataset,name+".csv")
	createTestDataFeatureCopy(featureFunction,name)

def createTestDataFeatureCopy(featureFunction,name):
	FILE_PATH = "rename/"
	testData = os.listdir(FILE_PATH)
	results = []
	first = True
	for file in testData:
		if first:
			row = featureFunction(readAUFile(FILE_PATH+file))
			indexList = ["id"] + range(0,len(row))
			results.append(indexList)
			results.append(np.append([file],row))
			first = False
		else:
			row = featureFunction(readAUFile(FILE_PATH+file))
			results.append(np.append([file],row))
	plainDataToCSV(results,name+"_TEST.csv")	

def genres():
	genres = {
		"metal":10,
		"classical":9,
		"blues":8,
		"country":7,
		"disco":6,
		"hiphop":5,
		"jazz":4,
		"pop":3,
		"reggae":2,
		"rock":1
	}
	return genres

def keyToGenre(key):
	reverseGenres = {v: k for k, v in genres().iteritems()}
	return reverseGenres[key]

def mapKeyToInt(key):
	ggenres = genres()
	return ggenres[key]

def loadValidationFS(featureSet,normalize=False,PCA=False,PCA_N=0):
	#result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
	dfset = []
	getClass = []
	for i in featureSet:
		df = readCSVToDF(i+".csv")
		getClass = df['Class']

		df = df.drop('Class',axis=1)
		dfset.append(df)
		
	df = pd.concat(dfset,axis=1)
	df.columns = range(0,len(pd.concat(dfset,axis=1).columns))

	if normalize:
		for i in df.columns.values:

	  		if i != 'Class':
	  			if np.std(df[i]) == 0:
	  				#raise Exception('One column has 0 STD in your features!!')
	  				#np.std(df[i]) = np.std(df[i])+0.0001
	  				df[i] = ( df[i] - sum(df[i])/len(df) ) / 0.0001
	  			else:	  				
	  				#df[i] = 1
	  				#print i,sum(df[i]),len(df)
	  				#average = sum(df[i])/len(df)
	  				#print len(df)
	  				#print average

	  				df[i] = ( df[i] - sum(df[i])/len(df) ) / (np.std(df[i]))

	if PCA:

		#df = df.fillna(method='backfill') FastICA
		pca = decomposition.PCA(n_components=PCA_N) # this looks to be a pretty sweet spot for SVM
		pca.fit(df)	
		df = pd.DataFrame(pca.transform(df))

	df['Class'] = getClass

	return df

def loadSubmissionFS(featureSet,normalize=False,PCA=False,PCA_N=0):
	dfset = []
	IDS = []
	for i in featureSet:
		df = readCSVToDF(i+"_TEST.csv")
		IDS = df['id']

		df = df.drop('id',axis=1)
		dfset.append(df)

		df = pd.concat(dfset,axis=1)
		df.columns = range(0,len(pd.concat(dfset,axis=1).columns))
		
		if normalize:
			for i in df.columns.values:
		  		if i != 'id':
		  			if np.std(df[i]) == 0:
		  				df[i] = ( df[i] - sum(df[i])/len(df) ) / 0.0001
		  				#raise Exception('One column has 0 STD in your features!!')
		  			else:	  				
		  				df[i] = ( df[i] - sum(df[i])/len(df) ) / (np.std(df[i]))

		if PCA:
			pca = decomposition.PCA(n_components=PCA_N) # this looks to be a pretty sweet spot for SVM
			pca.fit(df)	
			df = pd.DataFrame(pca.transform(df))

	df.insert(0, 'id', IDS)

	return df




