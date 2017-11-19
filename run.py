from features import *
from fileOperations import *
from validation import *
from classifier import *
import numpy as np
import math
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#  if true - build features
features = True

# if true - run set of validation tests 
validation = True

# if true - create a kaggle submission
kaggleSubmission = False


# program - this is the startup method which will run the options specified above
def program():
	# 661794 - frames per song 
	
	if features:
		createFeatures() 

	if validation:
		crossValidate()

	if kaggleSubmission:
		kaggle()



###### createFeatures - will run a present list of feature generators, and make a new set of CSV files for each feature

def createFeatures():

	createFeature(FFTComponents,'FFT')
	createFeature(createMFCC,'MFCC')
	createFeature(harmonicPercussiveSumStats,'HPsumStats')
	createFeature(spectralCentroidStats,'centroidStats')



######  crossValidate - will run a present list of tests with different settings 

def crossValidate():

	print "This script may take some time, after each test is run, a score will be output, and a confusion matrix will display"
	
	printTest(SVM,['MFCC'],True,False,1000)	
	printTest(LG,['MFCC'],True,False,1000)	

	return 

	# best result 
	printTest(SVM,['MFCC','centroidStats','HPsumStats'],True,False,1000)	
	printTest(LG,['MFCC','centroidStats','HPsumStats'],True,False,1000)	

	# all features 
	printTest(SVM,['FFT','MFCC','centroidStats','HPsumStats'],True,False,1000)	
	printTest(LG,['FFT','MFCC','centroidStats','HPsumStats'],True,False,1000)	
	
	# FFT and MFCC used 
	printTest(SVM,['FFT','MFCC'],True,False,1000)	
	printTest(LG,['FFT','MFCC'],True,False,1000)	
	
	# FTT + F3 
	printTest(SVM,['FFT','centroidStats','HPsumStats'],True,False,1000)	
	printTest(LG,['FFT','centroidStats','HPsumStats'],True,False,1000)	

	# FFT
	printTest(SVM,['FFT'],True,False,1000)	
	printTest(LG,['FFT'],True,False,1000)	

	# MFCC
	printTest(SVM,['MFCC'],True,False,1000)	
	printTest(LG,['MFCC'],True,False,1000)	

	# just F3 
	printTest(SVM,['centroidStats','HPsumStats'],True,False,1000)	
	printTest(LG,['centroidStats','HPsumStats'],True,False,1000)		



###### kaggle - will load all features specified and genereate a kaggle submission 
def kaggle():

	featuresSVM = ['justMFCC','centroidStats'] 	
	features = featuresSVM
	normalize = True
	PCA = False
	PCA_components = 2

	train = loadValidationFS(features,normalize,PCA,PCA_components)
	test = loadSubmissionFS(features,normalize,PCA,PCA_components)

	test["prediction"] = 0
	test = LG(train,test)
	predictionList = test['prediction'].tolist() #keyToGenre(test['prediction'])
	
	genres = []
	for k in predictionList:
		genres.append(keyToGenre(k))

	test['class'] = genres
	test = test[["id","class"]]

	test.to_csv('kaggleSubmission.csv',index = False)



###### printTest - method which governs how data is displayed when a validation test is run 
# 	classifier - the classifier to use for the test
# 	features - (list of strings) a list of features to be loaded to the classifier
# 	normalize - (boolean) true if data should be normailzed before being given to the classifier
# 	PCA - (boolean) true if PCA use be used to reduce dimensions in data 
#   PCA_components - (int) number of PCA dimensions to reduce to 

def printTest(classifier,features,normalize,PCA,PCA_components):
	bdata = loadValidationFS(features,normalize,PCA,PCA_components)
	print "FEATURES: - "+str(features)
	print "ACCURACY FROM "+str(classifier)
	foldsResult = kfoldsTest(classifier,10,bdata)
	a = map(keyToGenre,foldsResult[0]['truth'],) 
	b = map(keyToGenre,foldsResult[0]['predictions'])
	print foldsResult[1]
	plot_confusion_matrix(confusion_matrix(a,b),genres().keys(),normalize=True)
	plt.show()

	print ''

