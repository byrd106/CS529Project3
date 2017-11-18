from features import *
from fileOperations import *
from validation import *
from classifier import *
import numpy as np
import math
from sklearn import svm
from sklearn.decomposition import PCA

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


# creates a validation set for our features 
def createValidation():
	FILE_PATH = "rename/"
	testData = os.listdir(FILE_PATH)
	
	results = []
	first = True
	for file in testData:
		if first:
			row = twoFeatures(readAUFile(FILE_PATH+file))
			indexList = ["id"] + range(0,len(row))
			print file
			results.append(indexList)
			results.append(np.append([file],row))
			first = False
		else:
			row = twoFeatures(readAUFile(FILE_PATH+file))
			results.append(np.append([file],row))
	plainDataToCSV(results,"testTwoFeatures.csv")	


def createFeatures():

	#justMFCC, zeroCrossSumStats basicFFT zeroMFCC waveStats spectralRoll spectralRollStats
	#createFeature(RMSE,'RMSE')
	#createFeature(RMSEStats,'RMSEStats')
	#createFeature(zeroCrossingMFCC,'zeroMFCC')
	#createFeature(waveStats,'waveStats')
	#createFeature(harmonicPercussiveSumStats,'HPsumStats')
	#createFeature(spectralRollStats,'spectralRollStats') 
	#createFeature(chromaData,'chromaData') 
	#createFeature(getF3,'zeroCrossSumStats')
	#createFeature(harmonicPercussiveMFCC,'HPMFCC')
	#createFeature(spectralCentroidStats,'centroidStats')
	#createFeature(centroid,'centroid')
	#createFeature(flatChroma,'flatChroma')
	#createFeature(omfcc,'omfcc')
	
	#createFeature(smoothCrossZero,'smoothC')
	#createFeature(crossZero,'crossZR')
	
	createFeature(smoothCrossZero,'smoothC')
	createFeature(smoothCrossZero,'smoothC')

	print "HEHE"


def printTest(classifier,features,normalize,PCA,PCA_components):
	bdata = loadValidationFS(features,normalize,PCA,PCA_components)
	print "ACCURACY - "+str(classifier)
	foldsResult = kfoldsTest(classifier,10,bdata)
	a = map(keyToGenre,foldsResult[0]['truth'],) 
	b = map(keyToGenre,foldsResult[0]['predictions'])
	print foldsResult[1]
	plot_confusion_matrix(confusion_matrix(a,b),genres().keys(),normalize=True)
	plt.show()

	print ''


def crossValidate():

	#starting features 
	#basicData = ['justMFCC']
	#for i in range(1,1000,20):
		#printTest(SVM,['basicFFT'],True,True,i)	
	
	#printTest(LG,['basicFFT','centroidStats'],True,False,1000)	
	#printTest(SVM,['basicFFT','centroidStats'],True,False,1000)
	#,'HPsumStats','zeroCrossSumStats'

	#lets use HPSumStates and centroid stats 
	printTest(SVM,['justMFCC','centroidStats'],True,False,1000)	

	# creates confusion matricies and runs 10 fold validation: 
	
	#support vector machine, no normalization or PCA 
	#printTest(SVM,basicData,False,False,1000)	
	
	#logistic regression, no normalization or PCA
	#printTest(LG,basicData,False,False,1000)

	#support vector machine, no normalization or PCA 
	#printTest(SVM,basicData,True,False,1000)	

	#logistic regression, with normalization or PCA
	#printTest(LG,basicData,True,False,1000)


	#added centroid stats, third feature
	#'basicFFT',
	#newFeatures = ['justMFCC','centroidStats','HPsumStats','centroidStats'] #,'centroidStats'

	#support vector machine, no normalization or PCA 
	#printTest(SVM,newFeatures,False,False,1000)	
	
	#logistic regression, no normalization or PCA
	#printTest(LG,newFeatures,False,False,1000)

	#support vector machine, no normalization or PCA 
	#printTest(SVM,newFeatures,True,False,1000)	

	#logistic regression, with normalization or PCA
	#printTest(LG,newFeatures,True,False,1000)

	#optimal score 
	#newFeatures = ['justMFCC','centroidStats'] #  'HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats'] #,'centroidStats'
	#support vector machine, no normalization or PCA 
	#printTest(SVM,newFeatures,True,False,1000)	
	#printTest(LG,newFeatures,True,False,1000)	


	#SET WITH NORMALIZATION
	#printTest(SVM,basicData,True,False,1000) #.50 (way better than the SVM in this regard )
	#printTest(LG,basicData,True,False,1000)  #.35 (ALMOST A PERFECT HIPHOP!!!)
	# gets about a .50 

	#SET WITH NEW FEATURE? 

	# all seem to love centroidStats

	# basicData = ['justMFCC','centroidStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['justMFCC','centroidStats','HPsumStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['smoothC','crossZR','justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	# printTest(SVM,basicData,True,False,1000)

	#centroidStats - did LG Get worse with the new feature? 
	#,feature ... comes it at about 44~
	# regular LG... about 0.40 vs 0.44 

	#customFeatures = ['smoothC','crossZR','justMFCC','omfcc','tempo','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	
	#for feature in customFeatures:
	#	print feature
	#	printTest(LG,['justMFCC','basicFFT'],True,False,1000)	

	#newfeature = 
	#printTest(SVM,basicData,True,False,1000)
	#printTest(SVM,basicData,True,False,1000)

	#smoothC crossZR
	#collectiveFeatures = ['justMFCC','omfcc','tempo','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	#printTest(SVM,['justMFCC','basicFFT','HPsumStats','tempo','centroidStats'],True,True,1000)
	#printTest(GNB,['justMFCC','onsetStats','HPsumStats','zeroCrossSumStats','spectralRollStats'],True,True,38)
	#'justMFCC','basicFFT','HPsumStats','tempo','centroidStats'
	#printTest(SVM,['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,1000)
	#printTest(GNB,['justMFCC'],True,False,500) #,'HPsumStats'
	#printTest(LG,['smoothC','onsetStats','HPsumStats','zeroCrossSumStats','spectralRollStats'],True,False,500)


def kaggle():
	
	#train = readCSVToDF('onlyf2feature.csv')
	#secondSubmission 
	#loadValidationFS(['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,30)

	#featuresSVM = ['justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	featuresSVM = ['justMFCC','centroidStats'] 
	featuresGNB = ['spectralRollStats','RMSEStats','onsetStats','zeroCrossSumStats']
	featuresLG = ['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats']

	#basicData = ['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats']
	#printTest(LG,basicData,True,False,1000)
	
	features = featuresSVM

	normalize = True
	PCA = False
	PCA_components = 2

	train = loadValidationFS(features,normalize,PCA,PCA_components)
	test = loadSubmissionFS(features,normalize,PCA,PCA_components)
	
	print len(train)
	print len(test),test.shape
	#normalize these guys 
	
	#data = loadValidationFS(['zeroCrossSumStats','justMFCC','waveStats'],True,False,120)

	test["prediction"] = 0
	test = LG(train,test)
	#test = testClassifier(train,test)
	predictionList = test['prediction'].tolist() #keyToGenre(test['prediction'])
	
	genres = []
	for k in predictionList:
		genres.append(keyToGenre(k))

	test['class'] = genres
	test = test[["id","class"]]

	#print keyToGenre(list(test.iloc[3])[14])
	test.to_csv('LGl.csv',index = False)


def practice():

	# basicData = ['justMFCC','centroidStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['justMFCC','centroidStats','HPsumStats']
	# printTest(SVM,basicData,True,False,1000)

	# basicData = ['justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats']
	# printTest(SVM,basicData,True,False,1000)

	basicData = ['justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	printTest(GNB,basicData,True,False,1000)

	# basicData = ['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats']
	# printTest(SVM,basicData,True,False,1000)

	basicData = ['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats']
	printTest(LG,basicData,True,False,1000)


	#basicData = ['smoothC','crossZR','justMFCC','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	#printTest(SVM,basicData,True,False,1000)



	return 

	#centroidStats - did LG Get worse with the new feature? 
	#,feature ... comes it at about 44~
	# regular LG... about 0.40 vs 0.44 

	#customFeatures = ['smoothC','crossZR','justMFCC','omfcc','tempo','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	
	#for feature in customFeatures:
	#	print feature
	#	printTest(LG,['justMFCC','basicFFT'],True,False,1000)	

	#newfeature = 
	#printTest(SVM,basicData,True,False,1000)
	#printTest(SVM,basicData,True,False,1000)

	#smoothC crossZR
	#collectiveFeatures = ['justMFCC','omfcc','tempo','centroidStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	#printTest(SVM,['justMFCC','basicFFT','HPsumStats','tempo','centroidStats'],True,True,1000)
	#printTest(GNB,['justMFCC','HPsumStats'],True,False,500)
	#'justMFCC','basicFFT','HPsumStats','tempo','centroidStats'
	#printTest(SVM,['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,1000)
	#printTest(GNB,['justMFCC'],True,False,500) #,'HPsumStats'
	#printTest(LG,['smoothC','onsetStats','HPsumStats','zeroCrossSumStats','spectralRollStats'],True,False,500)



	#zeroMFCC might have helped naive beyes
	# bdata = loadValidationFS(collectiveFeatures,True,False,120)
	# #0.119637385087
	# classifier = GNB
	# print "ACCURACY - GNB"
	# foldsResult = kfoldsTest(classifier,10,bdata,True)
	# print ''
	#print confusion_matrix(foldsResult[0]['truth'],foldsResult[0]['predictions']),foldsResult[1]

	a = map(keyToGenre,foldsResult[0]['truth'],) 
	b = map(keyToGenre,foldsResult[0]['predictions'])

	plot_confusion_matrix(confusion_matrix(a,b),genres().keys())
	plt.show()
	
	data = loadValidationFS(collectiveFeatures,True,True,40)  
	classifier = SVM
	print "ACCURACY - SVM"
	print kfoldsTest(classifier,10,data)
	print ''

	return 

	classifier = LG
	print "ACCURACY - LG"
	ldata = loadValidationFS(collectiveFeatures,True,False,120)

	#print kfoldsTest(classifier,10,ldata)
	print ''
	print ''

	data = loadValidationFS(['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,120)
	#0.119637385087
	classifier = GNB
	print "BEST test ACCURACY - GNB"
	print kfoldsTest(classifier,10,data)	

	#BEST 
	data = loadValidationFS(['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,600)
	classifier = SVM
	print "BEST test ACCURACY - SVM"
	print kfoldsTest(classifier,10,data)
	print ''

	#BEST
	#data = loadValidationFS(['zeroCrossSumStats','justMFCC','waveStats'],True,False,120)
	data = loadValidationFS(['onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'],True,False,120)
	classifier = LG
	print "BEST test ACCURACY - LG"
	#print kfoldsTest(classifier,10,data)
	print ''
	print ''

	test = False

	##justMFCC, zeroCrossSumStats basicFFT zeroMFCC waveStats spectralRoll spectralRollStats
	if test:
		classifier = SVM
		for i in range(1,48,4):
			#'basicFFT','basicFFT'			
			print "SVM test'",i
			#featureSet = ['RMSEStats','spectralRoll','basicFFT','zeroMFCC','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats']
			#featureSet = ['zeroCrossSumStats']
			featureSet = ['justMFCC','RMSEStats','onsetStats','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats']
			data = loadValidationFS(featureSet,True,True,47) # True True 47 is doing better than the regular 
			print kfoldsTest(classifier,10,data)

			data = loadValidationFS(featureSet,True,True,i) # True True 47 is doing better than the regular 
			print kfoldsTest(classifier,10,data)

			#basicFFT


	print "Hey"
	# n_samples, n_features = 10, 5
	# y = np.random.randn(n_samples)
	# X = np.random.randn(n_samples, n_features)

	# clf = linear_model.Lasso(alpha=0.1)
	# clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	# print clf.predict([[2, 2],[1,1],[0,0]])

	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	pca = PCA(n_components=1)
	pca.fit(X) 
	X = pca.transform(X)
	print X

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


	#a = readCSVToDF('spectralRoll.csv')
	# justMFCC, zeroCrossSumStats basicFFT zeroMFCC waveStats spectralRoll spectralRollStats chroma data
	#'RMSE','RMSEStats','onsetStats','HPsumStats','zeroCrossSumStats','justMFCC','waveStats','spectralRollStats'
	#'justMFCC','justMFCC' spectralRollStats
	# spectralRollStats yes
	# onsetStats yes
	# waveStats meh
	# yes zeroCrossSumStats
	# 'HPsumStats'
	#'justMFCC',
	#collectiveFeatures = ['centroid','centroidStats','HPMFCC','justMFCC','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	#collectiveFeatures = ['flatChroma','centroidStats','justMFCC','HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats','onsetStats']
	#'justMFCC',
	#'flatChroma',
	#'HPMFCC',#


	# wow, with just MFCC...we reach 0.5 
	# this is really good HPsumStats + 1.5 vs wave stats 
	# the last 4 features aren't too powerful, we'll focus on HP sum stats
	#'HPsumStats','zeroCrossSumStats','waveStats','spectralRollStats'
	# HPsumStats is pretty good...spectralRollStats zero cross and regular wave are meh
	#zero cross/wave may be similar
	#,'justMFCC'
	#'onsetStats', small help
