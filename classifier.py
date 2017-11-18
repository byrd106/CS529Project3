import random 
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import decomposition

def noClass(train,test):
	return test

def testClassifier(train,test):

	for i, row in test.iterrows():
		test.loc[i, "prediction"] = random.choice(range(1,8))

 	return test

#def RNN 

#def SVM

def LASSOR(train,test):
	model = linear_model.Lasso(alpha=0.1)
	collength = train.shape[1]
	
	indexValues = train[map(str,range(0,collength-2))]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	lassoR = model.fit(trainFeatures,trainLabels)

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns
	Z = lassoR.predict(features)

	#print len(test),len(Z),'THIS IS A TEST'
	ints = []
	for i in Z:
		ints.append(int(round(i)))
	print ints
	test["prediction"] = ints

	return test


def GNB(train,test):
	#decision_function_shape='ovr',kernel='linear'
	model = GaussianNB()
	collength = train.shape[1]
	
	#indexFetch = map(str,range(0,collength-2))
	indexFetch = range(0,collength-2)
	
	indexValues = train[indexFetch]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	tSVM = model.fit(trainFeatures,trainLabels)

	features = test[indexFetch]	 # reduce to only feature columns
	Z = tSVM.predict(features)
	#print len(test),len(Z),'THIS IS A TEST'
	
	test["prediction"] = Z
	return test


def SVM(train,test):
	#decision_function_shape='ovr',kernel='linear'

	train = train.fillna(method='backfill')

	#model = svm.SVC(kernel='linear')

	model = svm.SVC(kernel='linear',C=1.0,decision_function_shape='ovr')
	collength = train.shape[1]
	
	#indexFetch = map(str,range(0,collength-2))
	indexFetch = range(0,collength-2)
	
	indexValues = train[indexFetch]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	tSVM = model.fit(trainFeatures,trainLabels)

	features = test[indexFetch]	 # reduce to only feature columns
	Z = tSVM.predict(features)
	#print len(test),len(Z),'THIS IS A TEST'
	
	test["prediction"] = Z
	return test


def RF(train,test):
	model = RandomForestClassifier
	collength = train.shape[1]
	
	indexValues = train[map(str,range(0,collength-2))]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	#logreg = model.fit(trainFeatures,trainLabels)

	X, y = make_classification(n_samples=len(trainFeatures), n_features=collength-1)
	model(X,y)

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns

	Z = model.predict(model,features)

	test["prediction"] = Z
	return test

def LN(train,test):

	linearModel = linear_model.LinearRegression()
	collength = train.shape[1]
	
	indexValues = train[map(str,range(0,collength-2))]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	logreg = linearModel.fit(trainFeatures,trainLabels)

	neigh = KNeighborsClassifier(n_neighbors=10)

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns
	Z = logreg.predict(features)

	#print len(test),len(Z),'THIS IS A TEST'
	
	test["prediction"] = Z
	return test


def KNN(train,test):
	#model = RandomForestClassifier()
	model =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	collength = train.shape[1]
	indexFetch = range(0,collength-2)
	indexValues = train[indexFetch]
	trainingSet = train["Class"]
	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	logreg = model.fit(trainFeatures,trainLabels)
	features = test[indexFetch]	 # reduce to only feature columns
	Z = logreg.predict(features)
	test["prediction"] = Z
	return test


def LG(train,test):
	train = train.fillna(method='backfill')

	model = linear_model.LogisticRegression(penalty='l2',C=1e25)
	collength = train.shape[1]
	indexFetch = range(0,collength-2)
	indexValues = train[indexFetch]
	trainingSet = train["Class"]
	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	logreg = model.fit(trainFeatures,trainLabels)
	features = test[indexFetch]	 # reduce to only feature columns
	Z = logreg.predict(features)
	test["prediction"] = Z
	return test

