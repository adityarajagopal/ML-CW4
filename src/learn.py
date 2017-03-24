import numpy as np
import data_manager as dm
from sklearn import svm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import decomposition

def cv(TrainMat, Y_hat, TestMat, Y, PCA, Prob4c):
	StartGamma = 0
	EndGamma = 0.02
	NumGamma = 10 
	GammaStep = (EndGamma - StartGamma) / float(NumGamma)
	GammaList = np.arange(StartGamma+GammaStep, EndGamma+0.000001, GammaStep)
	
	StartC = 0.0
	EndC = 10.0
	NumC = 20
	CStep = (EndC - StartC) / float(NumC)
	CList = np.arange(StartC+CStep, EndC+0.000001, CStep)
	
	if (PCA == 2):
		KList = [2]
	else:
		StartK = 0
		EndK = 100
		NumK = 100 
		KStep = int((EndK - StartK) / NumK)
		KList = [i for i in xrange(StartK+KStep, EndK+1, KStep)] 
	
	CVErrDict = dm.thread_cv(GammaList, CList, KList, TrainMat, Y_hat, PCA)

	if (PCA):
		X = TrainMat - np.mean(TrainMat, axis=0)
		TestSet = TestMat - np.mean(TestMat, axis=0)
		for K in CVErrDict:
			BestCG = min(CVErrDict[K], key=CVErrDict[K].get)
			PCAMat = pca(TrainMat, K)
			X = PCAMat.transform(TrainMat)
			TestSet = PCAMat.transform(TestMat)
			(TrainErr, TestErr, Svm_fit) = soft_margin_svm(X, Y_hat, TestSet, Y, BestCG[1], BestCG[0])
			CVErrDict.update({K:(BestCG, CVErrDict[K][BestCG], TrainErr, TestErr)})
		
		if (Prob4c):
			Ks = CVErrDict.keys()
			Values = CVErrDict.values()
			CGComb = [i[0] for i in Values]
			CVErrors = [i[1] for i in Values]
			TrainErrors = [i[2] for i in Values]
			TestErrors = [i[3] for i in Values]
			print "**************************************"
			print "Problem 4c (PCA)"
			for i in xrange(0,len(Ks)):
				print "K, C, G, CV, Train, Test: ", Ks[i], CGComb[i], CVErrors[i], TrainErrors[i], TestErrors[i]
			print "************************************** \n"
			
			ColourTest = ['b' if i == 1 else 'r' for i in Y]
			ColourTrain = ['b' if i == 1 else 'r' for i in Y_hat]

			X1 = TestSet[:,0]
			X2 = TestSet[:,1]
			X1min, X1max = X1.min()-1, X1.max()+1
			X2min, X2max = X2.min()-1, X2.max()+1
			xx, yy = np.meshgrid(np.linspace(X1min,X1max,100), np.linspace(X2min,X2max,100))
			Z = Svm_fit.decision_function(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			
			plt.figure()
			plt.title("PCA Test Set")
			plt.xlabel("X1")
			plt.ylabel("X2")
			plt.contour(xx, yy, Z, levels=[0])
			plt.scatter(X1, X2, c=ColourTest, s=1)
			plt.savefig("4c_2_test.eps", format='eps', dpi=1000)

			X1min, X1max = X.min()-1, X.max()+1
			X2min, X2max = X.min()-1, X.max()+1
			xx, yy = np.meshgrid(np.linspace(X1min,X1max,100), np.linspace(X2min,X2max,100))
			Z = Svm_fit.decision_function(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			
			plt.figure()
			plt.title("PCA Training Set")
			plt.xlabel("X1")
			plt.ylabel("X2")
			plt.contour(xx, yy, Z, levels=[0])
			plt.scatter(Svm_fit.support_vectors_[:,0], Svm_fit.support_vectors_[:,1], facecolor='none', edgecolor='black', s=20)
			plt.scatter(X[:,0], X[:,1], color=ColourTrain, s=1)
			plt.savefig("4c_2_train.eps", format='eps', dpi=1000)
			
		else:
			Ks = CVErrDict.keys()
			Values = CVErrDict.values()
			CGComb = [i[0] for i in Values]
			CVErrors = [i[1] for i in Values]
			TrainErrors = [i[2] for i in Values]
			TestErrors = [i[3] for i in Values]
			print "**************************************"
			print "Problem 4b"
			for i in xrange(0,len(Ks)):
				print "K, C, G, CV, Train, Test: ", Ks[i], CGComb[i], CVErrors[i], TrainErrors[i], TestErrors[i]
			print "************************************** \n"
			
			plt.figure()
			plt.title("Errors vs. k")
			plt.xlabel("k")
			plt.ylabel("Errors")
			plt.plot(Ks, CVErrors, c='r', label="CV Error")
			plt.plot(Ks, TrainErrors, c='g', label="Training Error")
			plt.plot(Ks, TestErrors, c='b', label="Test Error")
			plt.legend()
			plt.savefig("4b.eps", format='eps', dpi=1000)
	
	else:
		BestGamma = min(CVErrDict, key=CVErrDict.get)
		BestCVErr = CVErrDict[BestGamma]
		(Train_Err, Test_Err, Svm_fit) = hard_margin_svm(TrainMat, Y_hat, TestMat, Y, BestGamma)
		
		if (Prob4c):
			Gammas = CVErrDict.keys()
			CVErrors = CVErrDict.values()
			print "**************************************"
			print "Problem 4c (Features)"
			for i in xrange(0,len(Gammas)):
				print "Gamma, CVError: ", Gammas[i], CVErrors[i]
			print "BestGamma: %s, BestCVErr: %s" % (BestGamma, BestCVErr)
			print "Training, Test Errors: ", Train_Err, Test_Err
			print "************************************** \n"
			
			ColourTest = ['b' if i == 1 else 'r' for i in Y]
			ColourTrain = ['b' if i == 1 else 'r' for i in Y_hat]
			
			X1 = TestMat[:,0]
			X2 = TestMat[:,1]
			X1min, X1max = X1.min()-0.1, X1.max()+0.1
			X2min, X2max = X2.min()-0.1, X2.max()+0.1
			xx, yy = np.meshgrid(np.linspace(X1min,X1max,100), np.linspace(X2min,X2max,100))
			Z = Svm_fit.decision_function(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			
			plt.figure()
			plt.title("Feature Test Set")
			plt.xlabel("X1")
			plt.ylabel("X2")
			plt.contour(xx, yy, Z, levels=[0])
			plt.scatter(X1, X2, c=ColourTest, s=1)
			plt.savefig("4c_1_test.eps", format='eps', dpi=1000)
		
			X1min, X1max = TrainMat[:,0].min()-0.1, TrainMat[:,0].max()+0.1
			X2min, X2max = TrainMat[:,1].min()-0.1, TrainMat[:,1].max()+0.1
			xx, yy = np.meshgrid(np.linspace(X1min,X1max,100), np.linspace(X2min,X2max,100))
			Z = Svm_fit.decision_function(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			
			plt.figure()
			plt.title("Feature Training Set")
			plt.xlabel("X1")
			plt.ylabel("X2")
			plt.contour(xx, yy, Z, levels=[0])
			plt.scatter(Svm_fit.support_vectors_[:,0], Svm_fit.support_vectors_[:,1], facecolor='none', edgecolor='black', s=20)
			plt.scatter(TrainMat[:,0], TrainMat[:,1], color=ColourTrain, s=1)
			plt.savefig("4c_1_train.eps", format='eps', dpi=1000)
		
		else:
			Gammas = CVErrDict.keys()
			CVErrors = CVErrDict.values()
			print "**************************************"
			print "Problem 4a"
			for i in xrange(0,len(Gammas)):
				print "Gamma, CVerror: ", Gammas[i], CVErrors[i]
			print "BestGamma: %s, BestCVErr: %s" % (BestGamma, BestCVErr)
			print "Training, Test Errors: ", Train_Err, Test_Err
			print "************************************** \n"
	
			plt.figure()
			plt.title("CVError vs. Gamma")
			plt.xlabel("Gamma")
			plt.ylabel("CVError")
			plt.scatter(Gammas, CVErrors, color='red', s=20, label="CrossValidation Error")
			plt.legend()
			plt.savefig("4a.eps", format='eps', dpi=1000)


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
		(_, CVErr, _) = hard_margin_svm(TrainSet, TrainY, TestSet, TestY, Gamma)
		TotalCVErr += CVErr

	CVErr = TotalCVErr / (TrainMat.shape[0]/Fold)
	return CVErr

def n_fold_crossval_3((K, C, Gamma), Fold, TrainMat, Y_hat):
	TotalCVErr = 0
	TrainMat = TrainMat - np.mean(TrainMat, axis=0)		
	PCAMat = pca(TrainMat, K)
	TrainMat = PCAMat.transform(TrainMat)
	#print TrainMat.shape
	for Index in xrange(0, TrainMat.shape[0]-Fold, Fold):
		TrainSet = TrainMat
		TrainY = Y_hat
		for Row in xrange(Index, Index+Fold):
			TrainSet = np.delete(TrainSet, (Index), axis=0)
			TrainY = np.delete(TrainY, (Index), axis=0)
		TestSet = TrainMat[Index:Index+Fold,:]
		TestY = Y_hat[Index:Index+Fold,:]
		
		(_, CVErr, _) = soft_margin_svm(TrainSet, TrainY, TestSet, TestY, Gamma, C)
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
	
	return (Train_error, Test_error, rbf_svm)

def soft_margin_svm(X, Y_hat, TestSet, Y, Gamma, C):
	rbf_svm = svm.SVC(kernel='rbf', gamma=Gamma, shrinking=False, C=C)	
	rbf_svm = rbf_svm.fit(X, Y_hat.ravel())
	Z_train = rbf_svm.predict(X)
	Z_test = rbf_svm.predict(TestSet)

	E_train = Z_train+Y_hat.ravel()
	E_test = Z_test+Y.ravel()
	
	Train_error = np.sum([1 if val == 0 else 0 for val in E_train])/float(len(E_train))
	Test_error = np.sum([1 if val == 0 else 0 for val in E_test])/float(len(E_test))
	
	return (Train_error, Test_error, rbf_svm)

def pca(FeatMat, NumFeats):
	PCA = decomposition.PCA(n_components = NumFeats)
	Z = PCA.fit(FeatMat)
	return Z
