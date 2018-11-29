'''
Jarvis is a Bayes classifier.
comments on how to use him will be added later.
'''
import random as rd
import numpy as np
import math

class Jarvis:
	noOfClasses=0
	cov=[]
	mean=[]
	classifyMode=[]
	priorProb=[]
	confusionMatrix=[]
	accuracy=0.0
	precision=[]
	meanprecision=0.0
	meanrecall=0.0
	recall=[]
	fmeasure=[]
	ready=False

	def __init__(self):
		__init__(self, 0)

	def __init__(self, classes):
		self.noOfClasses = classes
		self.cov=[]
		self.mean=[]
		self.classifyMode=[]
		self.priorProb=[]
		self.confusionMatrix=[]
		self.ready=False

	def addPriorProb(self, priorProbability):
		self.priorProb.append(priorProbability)

	def setnoOfClasses(self, Classes):
		self.noOfClasses = classes

	def addCovMat(self, covarianceMatrix):
		self.cov.append(covarianceMatrix)

	def addMeanVect(self, meanVector):
		self.mean.append(meanVector)

	def setClassifyMode(self, classifyArray):
		self.classifyMode=classifyArray

	def isReady(self):
		if len(self.cov) == len(self.mean) == len(self.priorProb) == self.noOfClasses and len(self.classifyMode)!=0:
			self.ready = True
		else:
			self.ready = False
		return self.ready

	def discriminate2d(self, classNumber, xmat, ymat):
		ans = np.zeros((len(xmat), len(xmat[0])))
		for i in range(len(xmat)):
			for j in range(len(xmat[0])):
				ans[i][j] = self.gaussian(classNumber, [[xmat[i][j]],[ymat[i][j]]])
		return ans

	def gaussian(self, classNumber, point):
		#do the math here
		distance = np.subtract(point, self.mean[classNumber-1])
		sigma = self.cov[classNumber-1]
		sigmaInv = np.linalg.inv(sigma)
		ans = np.matmul(np.matmul(np.transpose(distance) , sigmaInv) , distance)
		ans = ans[0][0] #to convert from a matrix to a number
		ans += math.log(np.linalg.det(sigma))
		ans *= -0.5
		np.exp(ans)
		return ans

	def discriminate(self, classNumber, point):
		#do the math here
		distance = np.subtract(point, self.mean[classNumber-1])
		sigma = self.cov[classNumber-1]
		sigmaInv = np.linalg.inv(sigma)
		ans = np.matmul(np.matmul(np.transpose(distance) , sigmaInv) , distance)
		ans = ans[0][0] #to convert from a matrix to a number
		ans += math.log(np.linalg.det(sigma))
		ans *= -0.5
		ans += math.log(self.priorProb[classNumber-1])
		return ans

	def classify(self, point):
		if len(self.classifyMode) == 0:
			return -1
		bias = []
		for classNumber in self.classifyMode:
			bias.append(self.discriminate(classNumber, point))
		return self.classifyMode[bias.index(max(bias))]

	def convertToJarvis3(self):
		for i in range(len(self.cov)):
			self.cov[i][0][1]=0
			self.cov[i][1][0]=0

	def convertToJarvis2(self):
		matrix=self.cov[0]
		# accumulate
		for i in range(1,len(self.cov)):
			matrix+=self.cov[i]
		matrix = matrix/len(self.cov)
		for i in range(0,len(self.cov)):
			self.cov[i]=matrix

	def convertToJarvis1(self):
		self.convertToJarvis2()
		var = (self.cov[0][0][0]+self.cov[0][1][1])/2
		varmatrix = var*np.identity(2)
		for i in range(len(self.cov)):
			self.cov[i]=varmatrix
		
	def test(self, classNumber, point):
		self.confusionMatrix[self.classifyMode.index(classNumber)][self.classifyMode.index(self.classify(point))]+=1

	def initializeMetrics(self):
		self.accuracy=0.0
		self.meanprecision=0.0
		self.meanrecall=0.0
		self.recall=[]
		self.fmeasure=[]
		self.precision=[]
		dimension = len(self.classifyMode)
		self.confusionMatrix=np.zeros((dimension, dimension))
		
	def calculateMetrics(self):
		dimension = len(self.classifyMode)
		sumcols=np.sum(self.confusionMatrix, axis=0)
		sumrows=np.sum(self.confusionMatrix, axis=1)
		for i in range(dimension):
			self.recall.append(self.confusionMatrix[i][i]/sumrows[i])
			self.precision.append(self.confusionMatrix[i][i]/sumcols[i])
			self.fmeasure.append((2*self.precision[0]*self.recall[0])/(self.precision[0]+self.recall[0]))
			self.accuracy+=self.confusionMatrix[i][i]
		self.accuracy = self.accuracy / (sum(sumcols))
		self.meanrecall = np.mean(self.recall)
		self.meanprecision = np.mean(self.precision)

	def printMetrics(self):
		print('**** Metrics for mode ', str(self.classifyMode), ' ****')
		print('Recall = ', self.recall)
		print('Mean Recall = ', self.meanrecall)
		print('Precision', self.precision)
		print('Mean Precision = ', self.meanprecision)
		print('F Measure = ', self.fmeasure)
		print('Confusion Matrix = ', self.confusionMatrix)
		print('Accuracy = ', self.accuracy)

