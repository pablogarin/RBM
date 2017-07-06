#!/usr/bin/python
import numpy as np
from scipy import optimize
import sys
import random
'''
RBM	: Restricted Boltzman Machine.
Es un tipo de Red Neural Artificail (ANN) que descifra patrones por su cuenta sin supervision
'''
class RBM(object):
	def __init__(self, visibleSize=2, hiddenSize=3):
		self.visibleLayerSize = visibleSize
		self.hiddenLayerSize = hiddenSize
		# Learn rate should be around 10^-3
		#self.K = 0.001*(2./1.)
		#self.K = 0.001*(3./2.)
		#self.K = 0.001*(5./3.)
		#self.K = 0.001*(8./5.)
		#self.K = 0.001*(13./8.)
		#self.K = 0.001*(21./13.)
		self.K = 0.001*(34./21.)
		weights = 0.01 * np.random.randn(self.hiddenLayerSize, self.visibleLayerSize)
		#for i in xrange(hiddenSize):
		#	for j in xrange(visibleSize):
		#		value = random.uniform((0.1+self.K),0.4+self.K)
		#		weights[i][j] = value
		self.Weights = weights
		C = np.zeros((self.visibleLayerSize,))
		for n in xrange(self.visibleLayerSize):
			p = (n*1.)/(self.visibleLayerSize*1.)
			if p+self.K<1:
				p += self.K
			C[n] = np.log(p/(1.-p))
		self.visibleBias = C
		K = np.zeros((self.hiddenLayerSize,))
		self.hiddenBias = K

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))	

	def RELU(self, x):
		return max(0,x)
	
	def getParams(self):
		params = self.Weights.ravel()
		return params

	def setParams(self, params):
		hb, vb, w = params
		self.Weights = np.reshape(w, (self.hiddenLayerSize, self.visibleLayerSize))
		self.hiddenBias = np.reshape(hb, (self.hiddenLayerSize, ))
		self.visibleBias = np.reshape(vb, (self.visibleLayerSize, ))

	def check(self, X):
		h1 = self.computeProbability(value=X, bias=self.hiddenBias, initialSize=self.hiddenLayerSize, finalSize=self.visibleLayerSize)
		v2 = self.computeProbability(value=h1, bias=self.visibleBias, initialSize=self.visibleLayerSize, finalSize=self.hiddenLayerSize)
		return v2

	def computeProbability(self,value,bias,initialSize=0,finalSize=0):
		retval = np.zeros((initialSize,))

		# Check if Weights need to be rotated to fit Shape
		if len(self.Weights)<finalSize:
			weights = self.Weights
		else:
			weights = self.Weights.T

		for j in xrange(initialSize):
			sumation = 0
			for i in xrange(finalSize):
				sumation += value[i]*weights[j][i]
			retval[j] = self.sigmoid( bias[j] + sumation )
		return retval

	def trainNetwork(self,V, dbh, batch_size=10):
		'''
		Single-Step Contrastive Divergence (CD-1).
		'''
		CDSteps = 1
		c = 0
		total = len(V)
		for v in V:
			c += 1
			print "Elemento",c,"de",total
			
			# Gibbs Sampling
			v = v.reshape((self.visibleLayerSize,))
			h1 = self.computeProbability(value=v, bias=self.hiddenBias, initialSize=self.hiddenLayerSize, finalSize=self.visibleLayerSize)
			hf = h1
			for _ in xrange(CDSteps):
				v2 = self.computeProbability(value=hf, bias=self.visibleBias, initialSize=self.visibleLayerSize, finalSize=self.hiddenLayerSize)
				hf = self.computeProbability(value=v2, bias=self.hiddenBias, initialSize=self.hiddenLayerSize, finalSize=self.visibleLayerSize)

			# Get Deltas for Weights and Bias
			#rate = self.K*10.
			rate = self.K
			#deltaWeight = np.zeros(self.Weights.shape)
			#deltaHBias = np.zeros(self.hiddenBias.shape)
			#deltaVBias = np.zeros(self.visibleBias.shape)
			deltaWeight = rate*(np.outer(h1,v)-np.outer(hf,v2))
			deltaHBias = rate*(h1-hf)
			deltaVBias = rate*(v-v2)
			
			# Update Stage
			self.Weights += deltaWeight
			self.hiddenBias += deltaHBias
			self.visibleBias += deltaVBias 

			# Data Persistence
			if c%10==0:
				print "actualizando peso"
				upd = dbh.query("UPDATE weight SET weight=?, hiddenBias=?, visibleBias=? WHERE layer='Layer One';",(self.getParams(), self.hiddenBias.ravel(), self.visibleBias.ravel(),))
		print "actualizando peso"
		upd = dbh.query("UPDATE weight SET weight=?, hiddenBias=?, visibleBias=? WHERE layer='Layer One';",(self.getParams(), self.hiddenBias.ravel(), self.visibleBias.ravel(),))

	def bernoulli(self,p):
		b = np.random.rand(*p.shape) < p
		return np.array(b, dtype=float)
