import numpy as np
from sklearn import svm

def reader(File):
	Data = np.loadtxt(open(File,"rb"))
	return Data

def split_data(TrainMat, Val, Class):
	Train = np.zeros((1,TrainMat.shape[1]-1))
	
	for Row in TrainMat:
		if Row[0] == Val: 
			Train = np.append(Train, [Row[1:]], axis=0)
	Train = np.delete(Train, (0), axis=0)
	
	Y = Class * np.ones((Train.shape[0], 1))
	
	return (Train, Y)

def hard_margin_svm(X, Y_hat, TestSet, Y):
	rbf_svm = svm.SVC(kernel='rbf', gamma='auto', shrinking=False)	
	rbf_svm = rbf_svm.fit(X, Y_hat.ravel())
	Z_train = rbf_svm.predict(X)
	Z_test = rbf_svm.predict(TestSet)

	E_train = Z_train+Y_hat.ravel()
	E_test = Z_test+Y.ravel()
	
	Train_error = np.sum([1 if val == 0 else 0 for val in E_train])/float(len(E_train))
	Test_error = np.sum([1 if val == 0 else 0 for val in E_test])/float(len(E_test))
	print Train_error
	print Test_error

def main():
	TrainData = reader('../data/zip.train')
	TestData = reader('../data/zip.test')

	(Train2, Y_hat2) = split_data(TrainData, 2, 1)
	(Train8, Y_hat8) = split_data(TrainData, 8, -1)
	(Test2, Y2) = split_data(TestData, 2, 1)
	(Test8, Y8) = split_data(TestData, 8, -1)

	X = np.append(Train2, Train8, axis=0)
	Y_hat = np.append(Y_hat2, Y_hat8, axis=0)
	TestSet = np.append(Test2, Test8, axis=0)
	Y = np.append(Y2, Y8, axis=0)

	hard_margin_svm(X, Y_hat, TestSet, Y)


	

if __name__ == '__main__':
	main()
