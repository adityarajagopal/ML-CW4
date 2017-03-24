import numpy as np
import data_manager as dm
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import decomposition

def cv(TrainMat, Y_hat, TestMat, Y):
	StartGamma = 0.0
	EndGamma = 0.05
	NumGamma = 10 
	GammaStep = (EndGamma - StartGamma) / float(NumGamma)
	GammaList = np.arange(StartGamma+GammaStep, EndGamma+0.000001, GammaStep)
	
	StartC = 0.0
	EndC = 1.0
	NumC = 5
	CStep = (EndC - StartC) / float(NumC)
	CList = np.arange(StartC+CStep, EndC+0.000001, CStep)

	StartK = 0
	EndK = 256
	NumK = 5
	KStep = int((EndK - StartK) / NumK)
	KList = [i for i in xrange(StartK+KStep, EndK+1, KStep)] 
	
	PCA = False
	CVErrDict = dm.thread_cv(GammaList, CList, KList, TrainMat, Y_hat, PCA)

	if (PCA):
		X = TrainMat - np.mean(TrainMat, axis=0)
		TestSet = TestMat - np.mean(TestMat, axis=0)
		for K in CVErrDict:
			BestCG = min(CVErrDict[K], key=CVErrDict[K].get)
			PCAMat = pca(TrainMat, K)
			X = PCAMat.transform(TrainMat)
			TestSet = PCAMat.transform(TestMat)
			(TrainErr, TestErr) = soft_margin_svm(X, Y_hat, TestSet, Y, BestCG[1], BestCG[0])
			CVErrDict.update({K:(BestCG, CVErrDict[K][BestCG], TrainErr, TestErr)})
		
		print CVErrDict
		
		Ks = CVErrDict.keys()
		Values = CVErrDict.values()
		CVErrors = [i[1] for i in Values]
		TrainErrors = [i[2] for i in Values]
		TestErrors = [i[3] for i in Values]
		print CVErrors, TrainErrors, TestErrors
		plt.figure()
		plt.title("Errors vs. k")
		plt.xlabel("k")
		plt.ylabel("Errors")
		plt.scatter(Ks, CVErrors, c='r', s=1, label="CV Error")
		plt.scatter(Ks, TrainErrors, c='g', s=1, label="Training Error")
		plt.scatter(Ks, TestErrors, c='b', s=1, label="Test Error")
		plt.legend()
		plt.savefig("4b")
	
	else:
		BestGamma = min(CVErrDict, key=CVErrDict.get)
		BestCVErr = CVErrDict[BestGamma]
		print "BestGamma: %s, BestCVErr: %s" % (BestGamma, BestCVErr)
		
		(Train_Err, Test_Err) = hard_margin_svm(TrainMat, Y_hat, TestMat, Y, BestGamma)
		print Train_Err, Test_Err

		Gammas = CVErrDict.keys()
		CVErrors = CVErrDict.values()
		plt.figure()
		plt.title("Error vs. Gamma")
		plt.xlabel("Gamma")
		plt.ylabel("Error")
		plt.scatter(Gammas, CVErrors, c='r', s=1, label="Training Error")
		plt.legend()
		plt.savefig("4a")


def n_fold_crossval_1(Gamma, Fold, TrainMat, Y_hat):
	TotalCVErr = 0
	for Index in xrange(0, TrainMat.shape[0]-Fold, Fold):
		TrainSet = TrainMat
		TrainY = Y_hat
		for Row in xrange(Index, Index+Fold):
			TrainSet = np.delete(TrainSet, (Index), axis=0)
			TrainY = np.delete(TrainY, (Index), axis=0)
		TestSet = TrainMat[Index:Index+Fold,:]
		TestY = Y_hat[Index:Index+Fold,:]
		(_, CVErr) = hard_margin_svm(TrainSet, TrainY, TestSet, TestY, Gamma)
		TotalCVErr += CVErr

	CVErr = TotalCVErr / (TrainMat.shape[0]/Fold)
	return CVErr

def n_fold_crossval_3((K, C, Gamma), Fold, TrainMat, Y_hat):
	TotalCVErr = 0
	TrainMat = TrainMat - np.mean(TrainMat, axis=0)		
	PCAMat = pca(TrainMat, K)
	TrainMat = PCAMat.transform(TrainMat)
	print TrainMat.shape
	for Index in xrange(0, TrainMat.shape[0]-Fold, Fold):
		TrainSet = TrainMat
		TrainY = Y_hat
		for Row in xrange(Index, Index+Fold):
			TrainSet = np.delete(TrainSet, (Index), axis=0)
			TrainY = np.delete(TrainY, (Index), axis=0)
		TestSet = TrainMat[Index:Index+Fold,:]
		TestY = Y_hat[Index:Index+Fold,:]
		
		(_, CVErr) = soft_margin_svm(TrainSet, TrainY, TestSet, TestY, Gamma, C)
		TotalCVErr += CVErr

	CVErr = TotalCVErr / (TrainMat.shape[0]/Fold)
	return CVErr

def hard_margin_svm(X, Y_hat, TestSet, Y, Gamma):
	rbf_svm = svm.SVC(kernel='rbf', gamma=Gamma, shrinking=False)	
	rbf_svm = rbf_svm.fit(X, Y_hat.ravel())
	Z_train = rbf_svm.predict(X)
	Z_test = rbf_svm.predict(TestSet)

	E_train = Z_train+Y_hat.ravel()
	E_test = Z_test+Y.ravel()
	
	Train_error = np.sum([1 if val == 0 else 0 for val in E_train])/float(len(E_train))
	Test_error = np.sum([1 if val == 0 else 0 for val in E_test])/float(len(E_test))
	
	return (Train_error, Test_error)

def soft_margin_svm(X, Y_hat, TestSet, Y, Gamma, C):
	rbf_svm = svm.SVC(kernel='rbf', gamma=Gamma, shrinking=False, C=C)	
	rbf_svm = rbf_svm.fit(X, Y_hat.ravel())
	Z_train = rbf_svm.predict(X)
	Z_test = rbf_svm.predict(TestSet)

	E_train = Z_train+Y_hat.ravel()
	E_test = Z_test+Y.ravel()
	
	Train_error = np.sum([1 if val == 0 else 0 for val in E_train])/float(len(E_train))
	Test_error = np.sum([1 if val == 0 else 0 for val in E_test])/float(len(E_test))
	
	return (Train_error, Test_error)

def pca(FeatMat, NumFeats):
	PCA = decomposition.PCA(n_components = NumFeats)
	#Z = PCA.fit_transform(FeatMat)
	Z = PCA.fit(FeatMat)
	return Z
