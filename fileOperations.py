import csv
import os 
import pandas as pd
from fileOperations import *
from scikits.audiolab import Sndfile

###### readAUFile 
#	- reads an au file and returns data, given the path
def readAUFile(path):
	f = Sndfile(path, 'r')
	data = f.read_frames(f.nframes)
	return data


###### readCSVToDF 
#	- reads a CSV and returns a pandas dataframe 
#   -- csv - csv name
def readCSVToDF(csv):
	data = pd.read_csv(csv)
	return data


###### plainDataToCSV 
#	- takes the first 1000 components of a FFT transform of audio data and returns these as a feature
#   -- data - audio data of a single au file
def plainDataToCSV(datalist,name):
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		for row in datalist:
			wr.writerow(row)


###### listToCSV 
#	- takes a list of lists and creates a CSV from them 
#   -- datalist = list of lists 
#   -- name - name of csv
def listToCSV(datalist,name): 
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		indexValues = range(0,len(datalist[0])-1)
		indexValues.append("Class")
		wr.writerow(indexValues)
		for row in datalist:
			wr.writerow(row)


###### getTotalDataset 
#	- returns a dictionary with all file names across all genres, so the data processing systems can access these files
def getTotalDataset():
	fileTree = {} 
	genres = os.listdir("genres")
	genres.remove(".DS_Store")
	for genre in genres:
		if genre not in fileTree.keys():
			fileTree[genre] = []
		samples = os.listdir("genres/"+genre)
		samples.remove(".DS_Store")
		fileTree[genre] = samples
	return fileTree

