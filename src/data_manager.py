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
		if Row[0] == Val: 
			Train = np.append(Train, [Row[1:]], axis=0)
	Train = np.delete(Train, (0), axis=0)
	
	Y = Class * np.ones((Train.shape[0], 1))
	
	return (Train, Y)

def thread_gamma_cv(StartGamma, EndGamma, NumGamma, TrainMat, Y_hat):
	GammaStep = (EndGamma - StartGamma) / float(NumGamma)
	GammaList = np.arange(StartGamma+GammaStep, EndGamma+GammaStep, GammaStep)
	ProcessQ = Queue.Queue(len(GammaList))
	OpQ = Queue.Queue(len(GammaList))
	Threads = []
	ThreadNum = 10
	Fold = 10
	CVErrDict = {}

	for ThreadID in xrange(1, ThreadNum+1):
		Thread = CrossValThread(ThreadID, ProcessQ, OpQ)
		Thread.start()
		Threads.append(Thread)
	
	QueueLock.acquire()
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
		(CVParam, CVErr) = OpQ.get(block=False)
		CVErrDict.update({CVParam:CVErr})

	return CVErrDict


class CrossValThread(threading.Thread):
	def __init__(self, ThreadID, ProcessQ, OpQ):
		threading.Thread.__init__(self)
		self.ThreadID = ThreadID
		self.ProcessQ = ProcessQ
		self.OpQ = OpQ

	def run(self):
		#print "Starting Thread ", self.ThreadID
		process_data(self.ThreadID, self.ProcessQ, self.OpQ)
		print "Exiting ", self.ThreadID

def process_data(ThreadID, ProcQ, OpQ):
	while not ExitFlag:
		QueueLock.acquire()
		if not ProcQ.empty():
			(CVParam, Fold, TrainMat, Y_hat) = ProcQ.get()
			QueueLock.release()
			print "%s processing %s" % (ThreadID, CVParam)
			CVErr = learn.n_fold_crossval(CVParam, Fold, TrainMat, Y_hat)
			print "Gamma %s : CVErr = %s" % (CVParam, CVErr)
			QueueLock.acquire()
			OpQ.put((CVParam, CVErr))
			QueueLock.release()
		else:
			QueueLock.release()
	









