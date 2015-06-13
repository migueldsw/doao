import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter as ctr

def evaluateKNN(trainData, trainTarget, testData, testTarget):
	kList = [1,3,5,7,10,20,30]#1,3,5,7,10,20,30
	error = 100000000.0
	best_k = 0
	bestClassifier = []
	for k in kList:
		knn = KNeighborsClassifier(n_neighbors = k)
		knn.fit(trainData,trainTarget)
		error_temp = 0.0
		for i in range(len(testData)):
			if (knn.predict(testData[i])[0] != testTarget[i]): error_temp += 1.0
		if(error_temp < error):
			error = error_temp
			best_k = k
			bestClassifier = knn
	#print "KNN error = %f" %error
	#print "for k = %d" %best_k
	return error, best_k, bestClassifier

def evaluateDT(trainData, trainTarget, testData, testTarget):
	mLeafList = [1,2,3,5] # Min. datapoints  in a LEAF node
	mParentList = [5,10] #Min. datapoints  in a PARENT node
	prune = True #prune
	dtList = []#classifier list
	error = 100000000.0
	bestClassifier = []
	for l in mLeafList:
		for p in mParentList:
			dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=l,min_samples_split=p)
			dt.fit(trainData,trainTarget)
			dtList.append(dt)
	for dt in dtList:
		error_temp = 0.0
		for i in range(len(testData)):
			if (dt.predict(testData[i])[0] != testTarget[i]): error_temp += 1.0
		if(error_temp < error):
			error = error_temp
			bestClassifier = dt
	return error, bestClassifier

def random(X,Y):
	z = zip(X,Y)
	np.random.shuffle(z)
	newX, newY = [], []
	for i in z:
		newX.append(i[0])
		newY.append(i[1])
	return newX, newY

def get10Fold_n(X,Y,n):
	#get the n-th fold (of 10 folds) of X,Y as a test set, and all remaining folds as train set.
	testX = getFold(X,n)
	testY = getFold(Y,n)
	trainX = []
	trainY = []
	for i in range(10):
		if (i != n):
			trainX = trainX + getFold(X,i)
			trainY = trainY + getFold(Y,i)
	return trainX, trainY, testX, testY

def getFold(l,n):
	#get the n-th fold (starting from 0, of 10 folds {0,...,9}) from the list l
	return l[n*len(l)/10:(n+1)*len(l)/10]
	
def KFoldValidation(X,Y,classifierName):
##classifiers names:  ANN, DT, KNN, LDA, LR, SVM
	total_error = 0.0
	for i in range(10):
		a,b,c,d = get10Fold_n(X,Y,i)
		if classifierName == "KNN":
			error, bestk, bestClassifier = evaluateKNN(a,b,c,d )
		elif classifierName == "DT":
			error, bestClassifier = evaluateDT(a,b,c,d )
		total_error += error
	total_error = (float(total_error)/len(Y))*100
	#print "total error: %f %%" %total_error
	return total_error, bestClassifier

#TODO# def KFoldValidatin_CLASSIFIER2,3...(X,Y)

def OAOPairList(Y):#getOAOPairList
	#returns OneAgainstOne pairs list, for a given target list 'Y'
	classes = np.unique(Y)
	classes.sort()
	ret = []
	for c1 in classes:
		for c2 in classes:
			if (c1 < c2):
				ret.append((c1,c2))
	return ret

def OAODataSet(X,Y,OAOPair):#getOAODataSet
	#get a sub-dataset only with instance where the target is one from the given in the OAOPair pair
	newX, newY = [], []
	for i in range(len(Y)):
		if ((Y[i] == OAOPair[1]) | (Y[i] == OAOPair[0])):
			newX.append(X[i])
			newY.append(Y[i])
	return newX, newY

def OAO(X,Y,classifierName):
	pairs = OAOPairList(Y)
	bestError = 100
	bestPair = []
	classifierSet = []
	for p in pairs:
		#print "for classes: %d and %d:" %(p[0],p[1])
		nx, ny = OAODataSet(X,Y,p)
		error, best_knn = KFoldValidation(nx,ny,classifierName)
		#TODO# error2, bestclassifier2 = KFoldValidatin_CLASSIFIER2(nx,ny)
		#(find the best error)
		classifierSet.append(best_knn) #TODO# change to append the best classifier
		if (bestError > error):
			bestError = error
			bestPair = p
	print "\nBEST ERROR: %f %% | for class %d vs %d" %(bestError,bestPair[0],bestPair[1])
	return classifierSet

def DOAO(X,Y):
	pairs = OAOPairList(Y)
	bestError = 100
	bestPair = []
	classifierSet = []
	for p in pairs:
		#print "for classes: %d and %d:" %(p[0],p[1])
		nx, ny = OAODataSet(X,Y,p)
		error, best_knn = KFoldValidation(nx,ny,"KNN")
		#TODO# error2, bestclassifier2 = KFoldValidatin_CLASSIFIER2(nx,ny)
		#(find the best error)
		classifierSet.append(best_knn) #TODO# change to append the best classifier
		if (bestError > error):
			bestError = error
			bestPair = p
	print "\nBEST ERROR: %f %% | for class %d vs %d" %(bestError,bestPair[0],bestPair[1])
	return classifierSet

def OAOValidation(X,Y,classifierSet):
	##== OAO-KNN
	total_error = 0
	for i in range(10):
		trd,trt,ted,tet= get10Fold_n(X,Y,i)
		for c in classifierSet:
			c.fit(trd,trt)
		for j in range(len(ted)):
			votes = []
			for c in classifierSet:
				votes.append(c.predict(ted[j])[0])
			if (apureVotes(votes) != tet[j]): 
				total_error += 1
	return (float(total_error)/len(Y))*100

##TODO# def DOAOValidation(X,Y,classifierSet):
#sera igual, apenas no classifierSet pode vir diferentes classificadores

def apureVotes(li):
	return ctr(li).most_common(1)[0][0]

def OARValidation(X,Y,classifier):
	total_error = 0
	for i in range(10):
		trd,trt,ted,tet= get10Fold_n(X,Y,i)
		classifier.fit(trd,trt)
		for j in range(len(ted)):
			if (classifier.predict(ted[j])[0] != tet[j]): 
				total_error += 1
	return (float(total_error)/len(Y))*100

def OAO_OAR_KNN(X,y):
	e, knn = KFoldValidation(X,y,"KNN")
	print "-----Avaliando KNN para IRIS - KFOLD-cross validation (10 folds) OAO_OAR"
	oao = OAO(X,y,"KNN")
	print "-----KNN-OAO Validation Error:-----"
	print OAOValidation(X,y,oao)
	print "-----KNN-OAR Validation Error:-----"
	print OARValidation(X,y,knn)
def OAO_OAR_DT(X,y):
	e, knn = KFoldValidation(X,y,"DT")
	print "-----Avaliando DT para IRIS - KFOLD-cross validation (10 folds) OAO_OAR"
	oao = OAO(X,y,"DT")
	print "-----DT-OAO Validation Error:-----"
	print OAOValidation(X,y,oao)
	print "-----DT-OAR Validation Error:-----"
	print OARValidation(X,y,knn)


##EXECUTE#############
print "----MAIN----"
iris = datasets.load_iris()
X, y = iris.data, iris.target
X,y = random(X,y)
OAO_OAR_KNN(X,y)
OAO_OAR_DT(X,y)

