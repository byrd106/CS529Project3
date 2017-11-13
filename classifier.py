import random 
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def noClass(train,test):
	return test

def testClassifier(train,test):

	for i, row in test.iterrows():
		test.loc[i, "prediction"] = random.choice(range(0,12))

 	return test

#def RNN 

#def SVM

def SVM(train,test):
	model = svm.SVC(decision_function_shape='ovr',kernel='linear')
	collength = train.shape[1]
	
	indexValues = train[map(str,range(0,collength-2))]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	tSVM = model.fit(trainFeatures,trainLabels)

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns
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

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns
	Z = logreg.predict(features)

	#print len(test),len(Z),'THIS IS A TEST'
	
	test["prediction"] = Z
	return test

def LG(train,test):

	model = linear_model.LogisticRegression(penalty='l1',C=1e3)
	collength = train.shape[1]
	
	indexValues = train[map(str,range(0,collength-2))]
	trainingSet = train["Class"]

	trainFeatures = indexValues.values.tolist()
	trainLabels = trainingSet.values.tolist()
	logreg = model.fit(trainFeatures,trainLabels)

	features = test[map(str,range(0,collength-2))]	 # reduce to only feature columns
	Z = logreg.predict(features)

	#print len(test),len(Z),'THIS IS A TEST'
	
	test["prediction"] = Z
	return test

