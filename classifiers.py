import numpy as np
import time
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter as ctr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.lda import LDA
import neurolab as nl#
from datasets import DATA
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


ITERATIONS = 1
CLASSIFIERSLISTNAME = ["ANN", "DT","KNN","LDA","LR","SVM"]
DATASETNAMESLIST = ["zoo","iris","wine","seed","glass","ecoli","moviment","balance","landcover","vehicle","iris","vowel","yeast","car","segment"]
#CLASSIFIERSLISTNAME = ["KNN"] #just test
#DATASETNAMESLIST = ["ecoli","iris","car"] #just test
LOGLINES = []

def evaluateKNN(trainData, trainTarget, testData, testTarget):
	kList = [1,3,5,7,10,20,30]#1,3,5,7,10,20,30
	error = 1000000.0
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

def evaluateANN(trainData, trainTarget, testData, testTarget):
	#return evaluateLR(trainData[:len(trainData)], trainTarget[:len(trainData)], testData, testTarget)
	hiddenList=range(3,21)#no. hidden nodes 3~20  (3,21)
	#max iterations = 300
	error = 1000000.0
	bestClassifier = []
	for h in hiddenList:
		logistic = linear_model.LogisticRegression(C = 6000.0) #classifier to pipe
		rbm = BernoulliRBM(n_components=h,random_state=0,n_iter=30,learning_rate=0.05) #ann
		ann = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
		ann.fit(trainData,trainTarget)
		error_temp = 0.0
		for i in range(len(testData)):
			prediction = ann.predict(testData[i])[0]
			#print prediction
			if (prediction != testTarget[i]): error_temp += 1.0
		if(error_temp < error):
			error = error_temp
			bestClassifier = ann
	return error, bestClassifier

def evaluateDT(trainData, trainTarget, testData, testTarget):
	mLeafList = [1,2,3,5] # Min. datapoints  in a LEAF node
	mParentList = [5,10] #Min. datapoints  in a PARENT node
	prune = True #prune
	dtList = []#classifier list
	error = 1000000.0
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

def evaluateLR(trainData, trainTarget, testData, testTarget):
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
	error = 0.0
	lr = LogisticRegression()
	lr.fit(trainData,trainTarget)
	for i in range(len(testData)):
		if (lr.predict(testData[i])[0] != testTarget[i]): error += 1.0
	return error, lr

def evaluateLDA(trainData, trainTarget, testData, testTarget):
#http://scikit-learn.org/stable/auto_examples/classification/plot_lda.html#example-classification-plot-lda-py
	error = 0.0
	lda = LDA()
	lda.fit(trainData,trainTarget)
	for i in range(len(testData)):
		if (lda.predict(testData[i])[0] != testTarget[i]): error += 1.0
	return error, lda

def evaluateSVM(trainData, trainTarget, testData, testTarget):
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
	cList = [2**i for i in range (-3,11)] # param. C (SVM)
	sigmaList = [2**i for i in range (-5,6)] #param sigma (SVM)
	#cList=[1]
	#sigmaList=[0]
	#kernel -> (default='rbf') 
	svmList = []#classifier list
	error = 1000000.0
	bestClassifier = []
	for c in cList:
		for sigma in sigmaList:
			svm = SVC(C=c,gamma=sigma)
			svm.fit(trainData,trainTarget)
			svmList.append(svm)
	for svm in svmList:
		error_temp = 0.0
		for i in range(len(testData)):
			if (svm.predict(testData[i])[0] != testTarget[i]): error_temp += 1.0
		if(error_temp <= error): 
			error = error_temp
			bestClassifier = svm
	#print "SVM:: ERROR: %d" %error
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
		if (len(np.unique(b)) == 1) : b[0] = (np.unique(b)[0]+1)
		if classifierName == "KNN":
			error, bestk, bestClassifier = evaluateKNN(a,b,c,d )
		elif classifierName == "DT":
			error, bestClassifier = evaluateDT(a,b,c,d )
		elif classifierName == "LR":
			error, bestClassifier = evaluateLR(a,b,c,d )
		elif classifierName == "SVM":
			error, bestClassifier = evaluateSVM(a,b,c,d )
		elif classifierName == "LDA":
			error, bestClassifier = evaluateLDA(a,b,c,d )
		elif classifierName == "ANN":
			error, bestClassifier = evaluateANN(a,b,c,d )
		total_error += error
	total_error = (float(total_error)/len(Y))*100
	#print "total error: %f %%" %total_error
	return total_error, bestClassifier

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
		error, cls = KFoldValidation(nx,ny,classifierName)
		classifierSet.append(cls)
		if (bestError > error):
			bestError = error
			bestPair = p
	#print "\nBEST ERROR: %f %% | for class %d vs %d" %(bestError,bestPair[0],bestPair[1])
	return classifierSet

def OARClassList(Y): #returns a list with the classes in Y
	classes = np.unique(Y)
	classes.sort()
	return classes

def OARDataSet(X,Y,classe):
	newX, newY = [], []
	for i in range(len(Y)):
		if (Y[i] == classe):
			newX.append(X[i])
			newY.append(1)
		else:
			newX.append(X[i])
			newY.append(0)
	return newX, newY

def OAR(X,Y,classifierName):
	classes = OARClassList(Y)
	bestError = 100
	classifierSet = []
	for c in classes:
		#print "for class: %d:" %(c)
		nx, ny = OARDataSet(X,Y,c)
		error, cls = KFoldValidation(nx,ny,classifierName)
		classifierSet.append(cls)
		if (bestError > error):
			bestError = error
	return classifierSet

def DOAO(X,Y):
	if (len(np.unique(Y)) == 1) : Y[0] = (np.unique(Y)[0]+1)
	pairs = OAOPairList(Y)
	bestError = 100
	bestPair = []
	classifierSet = []
	classifierSetNames = []
	for p in pairs:
		#print "for classes: %d and %d:" %(p[0],p[1])
		classifierErrorList = []
		clfNameErrList = []
		nx, ny = OAODataSet(X,Y,p)
		for classifierName in CLASSIFIERSLISTNAME:
			error, classifier = KFoldValidation(nx,ny,classifierName)
			classifierErrorList.append((classifier,error))
			clfNameErrList.append((classifierName,error))
		classifierSet.append(takeMin(classifierErrorList))
		classifierSetNames.append(takeMin(clfNameErrList))
		print ("DOAO <- " + takeMin(clfNameErrList) + " pair: (%d,%d)") %(p[0],p[1])
		LOGLINES.append(("DOAO <- " + takeMin(clfNameErrList) + " pair: (%d,%d)") %(p[0],p[1]))
		#if (bestError > error):
		#	bestError = error
		#	bestPair = p
	#print "\nBEST ERROR: %f %% | for class %d vs %d" %(bestError,bestPair[0],bestPair[1])
	return classifierSet, classifierSetNames


def VOTE_Validation(X,Y):
	if (len(np.unique(Y)) == 1) : Y[0] = (np.unique(Y)[0]+1)
	pairs = OAOPairList(Y)
	total_error = 0
	for p in pairs:
		classifierSet = []
		#print "for classes: %d and %d:" %(p[0],p[1])
		nx, ny = OAODataSet(X,Y,p)
		for classifierName in CLASSIFIERSLISTNAME:
			error, classifier = KFoldValidation(nx,ny,classifierName)
			classifierSet.append(classifier)
		nx,ny = random(nx,ny)
		for i in range(10):
			trd,trt,ted,tet= get10Fold_n(nx,ny,i)
			for c in classifierSet:
				c.fit(trd,trt)
			for j in range(len(ted)):
				votes = []
				for c in classifierSet:
					votes.append(c.predict(ted[j])[0])
				if (apureVotes(votes) != tet[j]): 
					total_error += 1
	return (float(total_error)/len(Y))*100

def takeMin(tupleList):
	return min(tupleList, key = lambda t: t[1])[0]

def Validation(X,Y,classifierSet):
	##== OAO and OAR, by classifiers votes 
	if (len(np.unique(Y)) == 1) : Y[0] = (np.unique(Y)[0]+1)
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

def apureVotes(li):
	return ctr(li).most_common(1)[0][0]


def getTime():#in seconds
	return int(round(time.time() * 1000))

def resultsCompDataset(datasetName,numExecutions):
	(data,target) = DATA[datasetName]
	csvLineOut = ""
	sep = ","
	doaoClassSelect = []
	X = []
	Y = []
	for i in range(numExecutions):
		x,y = random(data,target)
		X.append(x)
		Y.append(y)
	for clName in CLASSIFIERSLISTNAME:
		#OAR
		errorTotal = 0
		for i in range(numExecutions):
			oar = OAR(X[i],Y[i],clName)
			er = Validation(X[i],Y[i],oar)
			errorTotal += er 
			#print "ERRO: %f" %er
		csvLineOut += "%.3f" %(float(errorTotal)/numExecutions) + sep
	for clName in CLASSIFIERSLISTNAME:
		#OAO
		errorTotal = 0
		for i in range(numExecutions):
			oao = OAO(X[i],Y[i],clName)  
			er = Validation(X[i],Y[i],oao)
			errorTotal += er 
			#print "ERRO: %f" %er
		csvLineOut += "%.3f" %(float(errorTotal)/numExecutions) + sep
	#VOTE-OAO
	errorTotal = 0
	for i in range(numExecutions):
		errorTotal += VOTE_Validation(X[i],Y[i])
		errorTotal += 0
	csvLineOut += "%.3f" %(float(errorTotal)/numExecutions) + sep
	#DOAO(proposed)
	errorTotal = 0
	for i in range(numExecutions):
		doao, clsNames = DOAO(X[i],Y[i])  
		er = Validation(X[i],Y[i],doao)
		errorTotal += er
		doaoClassSelect += clsNames
		#print "ERRO: %f" %er
	LOGLINES.append(printTupleList(avgSelection(doaoClassSelect,numExecutions)))
	print avgSelection(doaoClassSelect,numExecutions)
	csvLineOut += "%.3f\n" %(float(errorTotal)/numExecutions)
	#print csvLineOut
	return csvLineOut

def avgSelection(L,execs):
	items = np.unique(L)
	selections = []
	for it in items:
		c = 0.0
		for l in L:
			if (it == l ): c+=1.0
		selections.append((it,c/execs))
	return selections

def writeFileTable(datasetName,lines):
	f = open(datasetName+'_results.csv','w')
	for l in lines:
		f.write(l)
	f.close()
def writeFileLOG(datasetName,lines):
	f = open(datasetName+'_LOG-EXEC.txt','w')
	for l in lines:
		f.write(l + "\n")
	f.close()	

def printTupleList(list):
	str = ""
	for i in list:
		str += ("%s:%.2f  "%(i[0],i[1]))
	return str

def main():
	lines = []
	iterations = ITERATIONS
	totalTime = 0.0
	LOGLINES.append("itarations: %d"  %iterations)
	print "itarations: %d"  %iterations
	#for key, value in DATA.iteritems():
	for key in DATASETNAMESLIST:
		LOGLINES.append("-----------------")
		LOGLINES.append("Data set: " + key)
		print"-----------------"
		print "Data set: " + key
		st = getTime()
		line = resultsCompDataset(key,iterations)
		et = getTime()
		dt = float(et-st)/1000
		lines.append(line)
		totalTime += dt
		LOGLINES.append("in %f seconds" %dt)
		print "in %f seconds" %dt
		LOGLINES.append("-----------------")
		LOGLINES.append( "Total Time Execution(s): %.2f" %totalTime)
		print"-----------------"
		print "Total Time Execution(s): %.2f" %totalTime
		print "end"
		LOGLINES.append("---END---")
		writeFileTable(key,lines)
		writeFileLOG(key,LOGLINES)


##EXECUTE#############
LOGLINES.append("----DOAO TESTS EXECUTIONS----")
print "----DOAO TESTS EXECUTIONS----"
main()
#iris = datasets.load_iris()
#X, y = iris.data, iris.target
#(X,y) = DATA["zoo"]
#X, y = random(X,y)
#e, c = evaluateKNN(X,y,X[:8],y[:8])
#console run with pipe: "python classifiers.py > doao-log.txt 2>&1"
