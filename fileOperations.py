import csv
import os 
import pandas as pd


def readCSVToDF(csv):
	data = pd.read_csv(csv)
	return data

def plainDataToCSV(datalist,name):
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		for row in datalist:
			wr.writerow(row)

def listToCSV(datalist,name): 
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		indexValues = range(0,len(datalist[0])-1)
		#print indexValues,datalist[0]
		indexValues.append("Class")
		wr.writerow(indexValues)
		for row in datalist:
			wr.writerow(row)


def getTotalDataset(): # you may need to run these through a regex 
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