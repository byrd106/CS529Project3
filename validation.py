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


###### kfoldsTest - runs a k fold validation test
# 	classifier - the classifier to use for the test
# 	k - number of spits to use for k
# 	data - data frame of training set to split and test on 

def kfoldsTest(classifier,k,data):

	differenceResult = {}
	differenceResult["predictions"] = []
	differenceResult["truth"] = []

	normalize = True
	pca = True

	datalength = len(data)

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

		results.append(calculateAccuracy(test))
		differenceResult["predictions"] = differenceResult["predictions"] + list(predictions)
		differenceResult["truth"] = differenceResult["truth"] + list(truth)
		
	return [differenceResult,sum(results)/len(results)]



###### calculateAccuracy - calculate accuracy for a given training set and predictions
# 	data - dataframe of test set with predictions 

def calculateAccuracy(data):
		correct = float(len(data[data.prediction == data.Class]))
		wrong = float(len(data[data.prediction != data.Class]))
		return correct / (wrong + correct)



###### plotConfusionMatrix - plots a confusion matrix for a set of predictions and truth values
# 
# note, this method is a version of confusion matrix plot on this page:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#
# 	data - dataframe of test set with predictions 
# 	classes - the prediction classes
# 	normalized - normalize the values in the matrix 
# 	title - ti of the matrix 
# 	cmap - color map to use for the matrix 

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
