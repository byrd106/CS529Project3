from fileOperations import *
from classifier import *
import random
import itertools
import copy 
import math
from sklearn import svm
import numpy as np
from random import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

def kNormal(classifier,k,data):
	data = readCSVToDF(data)
	for i in data.columns.values:
		if i != 'Class':
			data[i] = sum(data[i])/len(data)
	return 2

def kfoldsTest(classifier,k,data,CM=False):

	differenceResult = {}
	differenceResult["predictions"] = []
	differenceResult["truth"] = []

	normalize = True
	pca = True

	datalength = len(data)

	print " ==== ==== ==== ==== ==== ==== "

	indexList = range(0,datalength)
	foldSize = datalength/k

	shuffle(indexList)
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

		predictions = test['prediction']
		truth = test['Class']

		#np.set_printoptions(precision=2)
		#plt.figure()
		#plot_confusion_matrix(confusion_matrix(truth,predictions),range(0,11), normalize=True)
		#plt.show()
		#heatmap( confusion_matrix(truth,predictions) , range(0,11))
		#print differenceResult["predictions"]
		#print differenceResult["predictions"],list(predictions),list(truth)

		results.append(calculateAccuracy(test))
		differenceResult["predictions"] = differenceResult["predictions"] + list(predictions)
		differenceResult["truth"] = differenceResult["truth"] + list(truth)
		
	return [differenceResult,sum(results)/len(results)]

def calculateAccuracy(data):
		correct = float(len(data[data.prediction == data.Class]))
		wrong = float(len(data[data.prediction != data.Class]))
		return correct / (wrong + correct)



# def heatmap(df, dict):
# 	fig, ax = plt.subplots()
# 	fig.subplots_adjust(left=0.3)
# 	#im = ax.imshow(df, interpolation='nearest', cmap=plt.cm.ocean)
# 	print fig


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #heatmap(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


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