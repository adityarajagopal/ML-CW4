import numpy as np
import time
import learn 
import data_manager as dm

def main():
	TrainData = dm.reader('../data/zip.train')
	TestData = dm.reader('../data/zip.test')
	TrainFeat = dm.reader('../data/features.train')
	TestFeat = dm.reader('../data/features.test')

	(Train2, Y_hat2) = dm.split_data(TrainData, 2, 1)
	(Train8, Y_hat8) = dm.split_data(TrainData, 8, -1)
	(Test2, Y2) = dm.split_data(TestData, 2, 1)
	(Test8, Y8) = dm.split_data(TestData, 8, -1)
	

	X = np.append(Train2, Train8, axis=0)
	Y_hat = np.append(Y_hat2, Y_hat8, axis=0)
	TestSet = np.append(Test2, Test8, axis=0)
	Y = np.append(Y2, Y8, axis=0)
	
	learn.cv(X, Y_hat, TestSet, Y)
			

if __name__ == '__main__':
	main()
