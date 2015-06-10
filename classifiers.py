import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


def evaluateKNN(trainData, trainTarget, testData, testTarget):
	kList = [1,3,5,7,10,20,30]#1,3,5,7,10,20,30
	error = 100.0
	best_k = 0
	for k in kList:
		knn = KNeighborsClassifier(n_neighbors = k)
		knn.fit(trainData,trainTarget)
		error_temp = 0.0
		for i in range(len(testData)):
			if (knn.predict(testData[i])[0] != testTarget[i]): error_temp += 1.0
		if(error_temp < error):
			error = error_temp
			best_k = k


	error = (error/len(testData))*100
	#print "KNN error = %f" %error
	#print "for k = %d" %best_k
	return error, best_k

def random(X,Y):
	z = zip(X,Y)
	np.random.shuffle(z)
	newX = []
	newY = []
	for i in z:
		newX.append(i[0])
		newY.append(i[1])
	return newX, newY

def get10Fold_n(X,Y,n):
	#get the n-th fold of X,Y as a test set, and all others folds as train set. of 10 folds
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
	 

def KFoldValidation_KNN(X,Y):
	error = 100.0
	for i in range(10):
		a,b,c,d = get10Fold_n(X,Y,i)
		erro, bestk = evaluateKNN(a,b,c,d )
		print "fold %d:  error:  %f  , best k: %d" %(i+1,erro,bestk)


#evaluateKNN(X[:-2], y[:-2],X[-2:], y[-2:])
#evaluateKNN(getFold(X,0),getFold(y,0),getFold(X,1),getFold(y,1))
#a,b,c,d = get10Fold_n(X,y,0)
#evaluateKNN(a,b,c,d )

print "-----Avaliando KNN para IRIS - KFOLD-cross validation (10 folds)"
iris = datasets.load_iris()
X, y = iris.data, iris.target
X,y = random(X,y)
KFoldValidation_KNN(X,y)



