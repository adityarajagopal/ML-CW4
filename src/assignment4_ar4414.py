import numpy as np
import time
import learn 
import data_manager as dm

def main():
	TrainData = dm.reader('../data/zip.train')
	TestData = dm.reader('../data/zip.test')
	TrainFeat = dm.reader('../data/features.train')
	TestFeat = dm.reader('../data/features.test')
	
	Xab, Y_hat_ab, TestSet_ab, Y_ab  = dm.gen_matrices([2], [8], TrainData, TestData, TrainFeat, TestFeat, 0)
	
	#PCA = 0
	#Prob4c = 0
	#learn.cv(Xab, Y_hat_ab, TestSet_ab, Y_ab, PCA, Prob4c)
	#print "a done"
	
	PCA = 1
	Prob4c = 0
	learn.cv(Xab, Y_hat_ab, TestSet_ab, Y_ab, PCA, Prob4c)
	print "b done"

	#Xc_raw, Y_hat_c_raw, TestSet_c_raw, Y_c_raw = dm.gen_matrices([1], [0,2,3,4,5,6,7,8,9], TrainData, TestData, TrainFeat, TestFeat, 0)
	#print "extracted data"
	#
	#PCA = 2
	#Prob4c = 1
	#learn.cv(Xc_raw, Y_hat_c_raw, TestSet_c_raw, Y_c_raw, PCA, Prob4c)

	#Xc_feat, Y_hat_c_feat, TestSet_c_feat, Y_c_feat = dm.gen_matrices([1], [0,2,3,4,5,6,7,8,9], TrainData, TestData, TrainFeat, TestFeat, 1)
	#print "extracted data"
	#
	#PCA = 256
	#Prob4c = 1
	#learn.cv(Xc_feat, Y_hat_c_feat, TestSet_c_feat, Y_c_feat, PCA, Prob4c)
	
			

if __name__ == '__main__':
	main()
