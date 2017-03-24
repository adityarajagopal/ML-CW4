import numpy as np
import Queue 
import threading
import learn
import time

ExitFlag = 0
QueueLock = threading.Lock()

def reader(File):
	Data = np.loadtxt(open(File,"rb"))
	return Data

def split_data(TrainMat, Val, Class):
	Train = np.zeros((1,TrainMat.shape[1]-1))
	
	for Row in TrainMat:
		if Row[0] in Val: 
			Train = np.append(Train, [Row[1:]], axis=0)
	Train = np.delete(Train, (0), axis=0)
	
	Y = Class * np.ones((Train.shape[0], 1))
	
	return (Train, Y)

def gen_matrices(Class1, Class2, TrainRaw, TestRaw, TrainFeat, TestFeat, Feat):
	if (Feat):
		TrainData = TrainFeat
		TestData = TestFeat
	else:
		TrainData = TrainRaw
		TestData = TestRaw

	(Train2, Y_hat2) = split_data(TrainData, Class1, 1)
	(Train8, Y_hat8) = split_data(TrainData, Class2, -1)
	(Test2, Y2) = split_data(TestData, Class1, 1)
	(Test8, Y8) = split_data(TestData, Class2, -1)

	X = np.append(Train2, Train8, axis=0)
	Y_hat = np.append(Y_hat2, Y_hat8, axis=0)
	TestSet = np.append(Test2, Test8, axis=0)
	Y = np.append(Y2, Y8, axis=0)

	return (X, Y_hat, TestSet, Y)
	

def thread_cv(GammaList, CList, KList, TrainMat, Y_hat, PCA):
	if (PCA):
		QLength = len(GammaList) * len(KList) * len(CList)
	else:
		QLength = len(GammaList)
	ProcessQ = Queue.Queue(QLength)
	OpQ = Queue.Queue(QLength)
	Threads = []
	ThreadNum = 8
	Fold = 100
	CVErrDict = {}

	for ThreadID in xrange(1, ThreadNum+1):
		Thread = CrossValThread(ThreadID, ProcessQ, OpQ, PCA)
		Thread.start()
		Threads.append(Thread)
	
	QueueLock.acquire()
	if (PCA):
		for k in KList:
			for c in CList:
				for g in GammaList:
					ProcessQ.put(((k, c, g), Fold, TrainMat, Y_hat))
	else:
		for g in GammaList:
			ProcessQ.put((g, Fold, TrainMat, Y_hat))

	QueueLock.release()

	start = time.time()
	while not ProcessQ.empty():
		pass 

	global ExitFlag
	ExitFlag = 1

	[T.join() for T in Threads]
	print "CV Time : %s" % (time.time() - start)
	
	while not OpQ.empty():
		if (PCA):
			((K, C, G), CVErr) = OpQ.get(block=False)
			CVParam = (C, G)
			Temp = CVErrDict.get(K, {})
			Temp.update({CVParam:CVErr})
			CVErrDict.update({K:Temp})
		else:
			(CVParam, CVErr) = OpQ.get(block=False)
			CVErrDict.update({CVParam:CVErr})
	
	ExitFlag = 0
	return CVErrDict


class CrossValThread(threading.Thread):
	def __init__(self, ThreadID, ProcessQ, OpQ, PCA):
		threading.Thread.__init__(self)
		self.ThreadID = ThreadID
		self.ProcessQ = ProcessQ
		self.OpQ = OpQ
		self.PCA = PCA

	def run(self):
		#print "Starting Thread ", self.ThreadID
		process_data(self.ThreadID, self.ProcessQ, self.OpQ, self.PCA)
		print "Exiting ", self.ThreadID

def process_data(ThreadID, ProcQ, OpQ, PCA):
	while not ExitFlag:
		QueueLock.acquire()
		if not ProcQ.empty():
			(CVParam, Fold, TrainMat, Y_hat) = ProcQ.get()
			QueueLock.release()
			print "%s processing %s" % (ThreadID, CVParam)
			if (PCA):
				CVErr = learn.n_fold_crossval_3(CVParam, Fold, TrainMat, Y_hat)
			else:
				CVErr = learn.n_fold_crossval_1(CVParam, Fold, TrainMat, Y_hat)
			#print "Gamma %s : CVErr = %s" % (CVParam, CVErr)
			QueueLock.acquire()
			OpQ.put((CVParam, CVErr))
			QueueLock.release()
		else:
			QueueLock.release()
	









