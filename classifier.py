import random 
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn import decomposition

###### testClassifier - test classifier to make sure validation methods work
# 	train - training data
#   test  - data to make predictions on 
def testClassifier(train,test):
	for i, row in test.iterrows():
		test.loc[i, "prediction"] = random.choice(range(1,8))
 	return test


###### GNB - Gaussian Naive Beyes classifier 
# 	train - training data
#   test  - data to make predictions on 
def GNB(train,test):
	model = GaussianNB()
	collength = train.shape[1]
	indexFetch = range(0,collength-2)
	indexValues = train[indexFetch]
	trainingSet = train["Class"]
	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	tSVM = model.fit(trainFeatures,trainLabels)
	features = test[indexFetch]	 # reduce to only feature columns
	Z = tSVM.predict(features)	
	test["prediction"] = Z
	return test


###### SVM - Support Vector Machine Classifier 
# 	train - training data
#   test  - data to make predictions on 
def SVM(train,test):
	train = train.fillna(method='backfill')
	model = svm.SVC(kernel='linear',C=1.0,decision_function_shape='ovr')
	collength = train.shape[1]
	indexFetch = range(0,collength-2)
	indexValues = train[indexFetch]
	trainingSet = train["Class"]
	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	tSVM = model.fit(trainFeatures,trainLabels)

	features = test[indexFetch]	 # reduce to only feature columns
	Z = tSVM.predict(features)
	test["prediction"] = Z
	return test


###### LG - Logisitic Regression Classifier
# 	train - training data
#   test  - data to make predictions on
def LG(train,test):
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

