import numpy as np
import data_manager as dm
from sklearn import svm

def gamma_cv(TrainMat, Y_hat, TestMat, Y):
	StartGamma = 0.0
	EndGamma = 0.1
	NumGamma = 10
	CVErrDict = dm.thread_gamma_cv(StartGamma, EndGamma, NumGamma, TrainMat, Y_hat)

	BestGamma = min(CVErrDict, key=CVErrDict.get)
	BestCVErr = CVErrDict[BestGamma]
	print "BestGamma: %s, BestCVErr: %s" % (BestGamma, BestCVErr)

	(Train_Err, Test_Err) = hard_margin_svm(TrainMat, Y_hat, TestMat, Y, BestGamma)
	print Train_Err, Test_Err

def n_fold_crossval(Gamma, Fold, TrainMat, Y_hat):
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
