from fileOperations import *
from classifier import *
import random
import itertools
import copy 
import math
from sklearn import svm
import numpy as np

def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

def kNormal(classifier,k,data):
	data = readCSVToDF(data)
	for i in data.columns.values:
		if i != 'Class':
			data[i] = sum(data[i])/len(data)
	return 2

def kfoldsTest(classifier,k,data):

	data = readCSVToDF(data)

	#zscore normalizer
	#print data.columns.values
	
	for i in data.columns.values:
	  	if i != 'Class':
	  		data[i] = ( data[i] - sum(data[i])/len(data) ) / np.std(data[i])

	datalength = len(data)
	
	print " ==== ==== ==== ==== ==== ==== "

	indexList = range(0,datalength)
	foldSize = datalength/k
	chunks = [indexList[x:x+foldSize] for x in xrange(0, datalength, foldSize)]

	results = []
	for chunk in chunks:		

		listCopy = copy.copy(chunks)
		listCopy.remove(chunk)
		
		trainIndex = reduce(lambda x,y: x+y, listCopy)
		testIndex = chunk

		data['prediction'] = 0

		train = data.loc[trainIndex]
		test = data.loc[testIndex]

		test = classifier(train,test)
		results.append(calculateAccuracy(test))
		
	print results
	return sum(results)/len(results)

def calculateAccuracy(data):
		correct = float(len(data[data.prediction == data.Class]))
		wrong = float(len(data[data.prediction != data.Class]))
		return correct / (wrong + correct)

	#print chunks
	#for p in itertools.combinations(chunks,2):
	#	print p

	# samples = set([])
	# lista = set(range(0,10))
	# while()
	# 	dataset = set(random.sample(lista,2))
	# 	samples = samples | dataset
	# 	lista = lista - dataset
	# print samples
	# print lista