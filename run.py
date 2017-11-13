from features import *
from fileOperations import *
from validation import *
from classifier import *
import numpy as np
import math
from sklearn import svm
# librosa is required , 
# pip install librosa

# for logisitc regression 
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
#brew install libsndfile
#pip install scikits.audiolab

features = False
validation = True
kaggleSubmission = False
practiceS = False

testFile = '/Users/danielbyrd/Desktop/AppDevelopment/class/ML3/classical.00000.wav'
test2File = '/Users/danielbyrd/Desktop/AppDevelopment/class/ML3/genres/classical/classical.00001.au'

def program():
	# 661794 - frames per song 
	
	if features:
		createFeatures() 

	if validation:
		crossValidate()

	if kaggleSubmission:
		kaggle()
	
	if practiceS:
		practice()

	# now we need to do this^ for ever file, create a small CV system, and see if we get results\
	#print data.shape
	#print fs
	# and we also need first FFT 
	# we need 13 features per song, so average 13 coefficients over entire 4135 frames
	# mfcc
	#read out frames 
	#print 0.05 * np.random.randn(2, 48000)
	##a = np.random.randn(2, 10*48000)
	##play(0.05 * a)
	#play(0.05 * np.random.randn(2, 48000))

def genres():
	genres = {
		"metal":11,
		"blues":10,
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

# creates a validation set for our features 
def createValidation():
	FILE_PATH = "rename/"
	testData = os.listdir(FILE_PATH)
	
	results = []
	first = True
	for file in testData:
		if first:
			row = getF2(readAUFile(FILE_PATH+file))
			indexList = ["id"] + range(0,len(row))
			print file
			results.append(indexList)
			results.append(np.append([file],row))
			first = False
		else:
			row = getF2(readAUFile(FILE_PATH+file))
			results.append(np.append([file],row))
	plainDataToCSV(results,"testFeatures.csv")	



def createFeatures():
	#createValidation()
	#return 

	FILE_PATH = "genres/"
	outputDataset = []
	dataset = getTotalDataset()
	count = 0
	for key in dataset.keys():
		for row in dataset[key]:
			
			data = [666,667]
			
			if count not in data:
				
				#print FILE_PATH+key+"/"+row
				row = np.append(allFeatures(readAUFile(FILE_PATH+key+"/"+row)),mapKeyToInt(key))
				if math.isnan(float(row[1])):
					print count
				else:
					outputDataset.append(row)
			count+=1
			#print count

			#row = np.append(getF2(readAUFile(FILE_PATH+key+"/"+row)),mapKeyToInt(key))
			#outputDataset.append(row)
	
	#print len(outputDataset)
	listToCSV(outputDataset,'allFeatures.csv')


def crossValidate():
	#data = "onlyf2feature.csv"
	data = "allFeatures.csv"
	#data = "f2feature.csv"
	#classifier = testClassifier
	
	classifier = LG
	#classifier = SVM

	#classifier = RF
	#classifier = LN

	#print "ACCURACY",kfoldsTest(classifier,10,data)
	print "ACCURACY",kfoldsTest(classifier,10,data)

def kaggle():
	#createValidation()
	train = readCSVToDF('onlyf2feature.csv')
	test = readCSVToDF('testFeatures.csv')

	#normalize these guys 

	for i in train.columns.values:
	  	if i != 'Class':
	  		train[i] = ( train[i] - sum(train[i])/len(train) ) / np.std(train[i])
	
	for i in test.columns.values:
	  	if i != 'id':
	  		test[i] = ( test[i] - sum(test[i])/len(test) ) / np.std(test[i])
	
	test["prediction"] = 0
	test = LG(train,test)
	predictionList = test['prediction'].tolist() #keyToGenre(test['prediction'])
	
	genres = []
	for k in predictionList:
		genres.append(keyToGenre(k))


	test['class'] = genres
	test = test[["id","class"]]

	print test

	#print keyToGenre(list(test.iloc[3])[14])
	test.to_csv('kaggleSub.csv',index = False)


def practice():
	print "Hey"
	#Classes	3
	#Samples per class	50
	#Samples total	150
	#Dimensionality	4
	#Features	real, positive
	
	#clf = svm.SVC()
	#clf.fit(X, y)  

	# print "HEY!"
	# X = [[0,0,0], [2,1000,2] ,[2,2,2], [3,3,3]]
	# y = [0, 1, 2, 3]
	# clf = svm.SVC()
	# clf.fit(X, y)  
	# print clf.predict([[2, 20000, 2]])
	# clf = svm.SVC(decision_function_shape='ovo')
	# clf.fit(X, Y) 
	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)
	# dec = clf.decision_function([[1]])
	# data = [[2,2]]
	#print data.shape,"Ehe"
	#print clf.predict(data)

	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)

	#data = "f2feature.csv"
	#data = readCSVToDF(data)
	# print data
	# count = 0
	# for row in data.values:
	# 	print count,row
	# 	count+=1
	#print count



	# iris = datasets.load_iris()
	# X = iris.data[:, :2]  # we only take the first two features.
	# Y = iris.target

	#	print X.shape
	#	print Y.shape

	# for i in range(0,100):
	# 	print " ======= "
	# 	print X[i]
	# 	print Y[i]

	# logreg = linear_model.LogisticRegression(C=1e5)
	# logreg.fit(X,Y)
	# data = [[5.5,2.6],[6.5,2.8],[4.6,3.1]]
	# Z = logreg.predict(data)

	#print Z

	#takes list of features, fits to list of corresponding train

	#Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])


