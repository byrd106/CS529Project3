import math
import os
import scikits.audiolab
import numpy as np
import scipy
import scipy.io.wavfile
import librosa
from fileOperations import *
import random
from sklearn import decomposition
from scipy import signal
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

###### FFTComponents 
#	- takes the first 1000 components of a FFT transform of audio data and returns these as a feature
#   -- data - audio data of a single au file
def FFTComponents(data):
	data = abs(scipy.fft(data)[:1000])
	return data

###### MFCC 
#	- gets the mfccs of audio data, and averages these across frames and returns the range from 10% to 90%
#   -- data - audio data of a single au file
def MFCC(data):
	ceps, mspec, spec = mfcc(data)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

###### spectralCentroidStats 
#	- gets the max,min,mean,std and var of the spectral centroid of audio data
#   -- data - audio data of a single au file
def spectralCentroidStats(data):
	lz = librosa.feature.spectral_centroid(data)[0]
	return [np.max(lz),np.std(lz),np.mean(lz),np.min(lz),np.var(lz)]

###### harmonicPercussiveSumStats 
#	- gets the max,min,mean,std and var of the separated harmonic and percussive components
#   -- data - audio data of a single au file
def harmonicPercussiveSumStats(data):
	y_harmonic, y_percussive = librosa.effects.hpss(data)
	summaryStats = [ 
			np.max(y_harmonic),np.std(y_harmonic),np.mean(y_harmonic),np.min(y_harmonic),np.var(y_harmonic),
			np.max(y_percussive),np.std(y_percussive),np.mean(y_percussive),np.min(y_percussive),np.var(y_percussive) 
	]
	return summaryStats


###### createFeature 
#	- takes a feature function , and creates a version of the data set w/ just this feature and writes this to CSV
#   and also takes the validation data, and makes a CSV as well.
#	This is done so multiple features can be generated and tested in differernt combinations without having to regenerate the data 
#	
#   featureFunction - a function which will genreate the desired feature
#	name - a designation for this feature, which is how it will be called in validation experiments, and the name of the file this outputs
def createFeature(featureFunction,name):
	FILE_PATH = "genres/"
	outputDataset = []
	dataset = getTotalDataset()
	count = 0
	for key in dataset.keys():
		for row in dataset[key]:
			data = [666,667] # these 2 files seem to have an issue w/ their data so we'll skip them for now 
			if count not in data:
				row = np.append(featureFunction(readAUFile(FILE_PATH+key+"/"+row)),mapKeyToInt(key))
				if math.isnan(float(row[1])):
					print count
				else:
					outputDataset.append(row)
			count+=1
	listToCSV(outputDataset,name+".csv")
	createTestDataFeatureCopy(featureFunction,name)


###### createTestDataFeatureCopy 
#	- takes a feature function , and creates a version of the data set w/ just this feature and writes this to CSV
#   for the test data
#	This is done so feature is ready to go to create a submission, each feature when doing experiments only ever needs to be generated once
#	
#   featureFunction - a function which will genreate the desired feature
#	name - a designation for this feature, which is how it will be called in validation experiments, and the name of the file this outputs
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


###### genres - dictionary of genres for key mapping 
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

###### keyToGenre - given a numeric key, get the genre from genres dictionary
def keyToGenre(key):
	reverseGenres = {v: k for k, v in genres().iteritems()}
	return reverseGenres[key]

###### mapGenreToInt - given a genre, get a numeric value for this genre from the genres dictionary
def mapGenreToInt(key):
	genreData = genres()
	return genresData[key]


###### loadValidationFS 
#	- combines features from several feature sets (as csvs) ,
#	to create a dataset w/ all of the features for classification and preprocesses these if needed
#	
#   featureSet - list of features to load and combine for classification / training
#	normalize - normalize the data 
#	PCA - use PCA
# 	PCA_N - number of PCA components to create 
def loadValidationFS(featureSet,normalize=False,PCA=False,PCA_N=0):
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
	  			df[i] = ( df[i] - sum(df[i])/len(df) ) / (np.std(df[i]))

	if PCA:

		pca = decomposition.PCA(n_components=PCA_N) # this looks to be a pretty sweet spot for SVM
		pca.fit(df)	
		df = pd.DataFrame(pca.transform(df))

	df['Class'] = getClass

	return df


###### loadSubmissionFS 
#	- combines features from several feature sets (as csvs) , (but test CSVs)
#	to create a dataset w/ all of the features for classification and preprocesses these if needed
#	
#   featureSet - list of features to load and combine for classification / training
#	normalize - normalize the data 
#	PCA - use PCA
# 	PCA_N - number of PCA components to create 
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
		  			df[i] = ( df[i] - sum(df[i])/len(df) ) / (np.std(df[i]))

		if PCA:
			pca = decomposition.PCA(n_components=PCA_N) # this looks to be a pretty sweet spot for SVM
			pca.fit(df)	
			df = pd.DataFrame(pca.transform(df))

	df.insert(0, 'id', IDS)

	return df


######################## NOTE ####################################
# the following methods were used to test various features but aren't referenced elsewhere due
# to their experimental nature, features which improved classifer and were part of the report
# are at the top of this file 

##################################################################


###### crossZero 
#	- gets the first 1100 data points of the zero crossing rate for an audio file 
#   -- data - audio data of a single au file
def crossZero(data):
	zcr = librosa.feature.zero_crossing_rate(data)[0][:1100]
	return zcr


###### smoothCrossZero 
#	- gets the first 1100 data points of the zero crossing rate for an audio file and uses a hann filter to smooth the wave
#   -- data - audio data of a single au file
def smoothCrossZero(data):
	win = signal.hann(50)
	zcr = librosa.feature.zero_crossing_rate(data)[0][:1100]
	sconvoler = signal.convolve(zcr, win, mode='same') / sum(win)
	return sconvoler


###### omfcc 
#	- gets omfcc for an audio file
#   -- data - audio data of a single au file
def omfcc(data):
	#mfcc_feat = mfcc(data,22050)
	fbank_feat = logfbank(data,22050)
	return fbank_feat[1:2,:][0]


###### centroid 
#	- gets 1000 data points of the spectral centroid of an audio file
#   -- data - audio data of a single au file
def centroid(data):
	return librosa.feature.spectral_centroid(data)[0][:1000]


###### waveStats 
#	- gives basic summary stats of audio wave 
#   -- data - audio data of a single au file
def waveStats(data):
	summaryStats = [np.max(data),np.std(data),np.mean(data),np.min(data),np.var(data)]
	return summaryStats


###### zeroCrossingMFCC 
#	- calculates the MFCCs of the zero crossing rate of an audio file (averages across frames and takes 10% to 90% section)
#   -- data - audio data of a single au file
def zeroCrossingMFCC(data):
	k = librosa.feature.zero_crossing_rate(data)[0]
	ceps, mspec, spec = mfcc(k)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x


###### spectralRoll 
#	- calculates the spectral rolloff of audio data and gives first 1000 components
#   -- data - audio data of a single au file
def spectralRoll(data):
	return librosa.feature.spectral_rolloff(data)[0][:1000]


###### spectralRoll 
#	- gets the summary statistics of the spectral rolloff of an audio file 
#   -- data - audio data of a single au file
def spectralRollStats(data):
	sk = librosa.feature.spectral_rolloff(data)[0]
	return [np.max(sk),np.std(sk),np.mean(sk),np.min(sk),np.var(sk)]


###### spectralRoll 
#	- gets the summary statistics of the spectral rolloff of an audio file 
#   -- data - audio data of a single au file
def onsetStats(data):
	sr = 22050
	onset_env = librosa.onset.onset_strength(y=data, sr=sr,aggregate=np.median, fmax=8000, n_mels=256)
	return [np.mean(onset_env),np.var(onset_env),np.std(onset_env),np.max(onset_env),np.min(onset_env)]


###### tempo 
#	- gives tempo of audio data
#   -- data - audio data of a single au file
def tempo(data):
	sr = 22050
	y_harmonic, y_percussive = librosa.effects.hpss(data)
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,	                                             sr=sr)
	return tempo





