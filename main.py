#!/usr/bin/python
import os, sys, sqlite3, ast
import math
import numpy as np
from RBM import RBM
from SQLHelper import SQLHelper
from mnist import *
from PIL import Image, ImageFilter
from random import randint

from scipy.interpolate import interp1d
VISUALIZE = False

dbh = None

'''
TODO: 
	los datos a analizar deben ser descompuestos en vectores del mismo tamano. 
	Para esto vamos a decidir un tamano fijo y modificar los datos dependiendo de su origen:
		- Audio: Muestreo a intervalos regulares, ej: transformar senal analoga en digital de 1000 de resolucion.
		- Imagen: Cambiar el tamano de las imagenes por uno fijo, rellenando la diferencia con blanco.
		- Texto: Generar un lexicon con palabras claves a buscar para determinar un bitmap de una frase a partir de las coincidencias. (ver NLTK).

	En caso de las imagenes, se puede usar una "Convoluted ANN", donde la imagen se divide en varos pedazos y cada capa analiza una parte. 
'''
	
def main(args):
	dbh = SQLHelper("database.sqlite")
	rbm = RBM(visibleSize=784, hiddenSize=500)
	
	cur = dbh.query("select * from weight where current=1;")
	if len(cur)>0:
		if cur[0]['weight'] is not None:
			params = cur[0]['hiddenBias'] ,cur[0]['visibleBias'], cur[0]['weight']
			rbm.setParams(params)
		
	if len(sys.argv)>1:
		for arg in sys.argv:
			if arg=='-t':
				testSet =  getTestSet(digits=[2])
				if VISUALIZE:
					# n = X[0] > np.random.rand(*X[0].shape)
					# n = n.reshape(28,28)
					n = testSet[0].reshape(28,28)
					strline = ""
					for l in n:
						for b in l:
							if b:
								strline += "*"
							else:
								strline += " "
						strline += "\n"
					print strline
					sys.exit()
				rbm.trainNetwork(testSet, dbh, 100)
				# for i in xrange(10):
				# 	testSet =  getTestSet(digits=[i])
				# 	if VISUALIZE:
				# 		n = X[0] > np.random.rand(*X[0].shape)
				# 		n = n.reshape(28,28)
				# 		strline = ""
				# 		for l in n:
				# 			for b in l:
				# 				if b:
				# 					strline += "*"
				# 				else:
				# 					strline += " "
				# 			strline += "\n"
				# 		print strline
				# 		sys.exit()
				# 	rbm.trainNetwork(testSet, dbh, 100)
				return
			if arg=='-w':
				#sqlite reg 830
				i = 0
				for weight in rbm.Weights:
					retval = rbm.sigmoid(weight.copy())
					saveImage((retval*255), "images/weight-"+str(i)+".png")
					i += 1
				#testSet =  getTestSet(randint(0,9))
				for i in range(0,10):
					testSet =  getTestSet(digits=[i], Bernoulli=False, dataSet='testing')
					retval = testSet[random.randint(0,len(testSet))].copy()
					saveImage(retval, "images/original-"+str(i)+".png")
					retval = np.array((np.random.rand(*retval.shape) < retval), dtype=float)
					retval = rbm.check(retval)
					retval = retval.reshape(rbm.visibleLayerSize)
					saveImage((255*retval), "images/result-"+str(i)+".png")
				return
	return 0

def getTestSet(digits=np.arange(10), Bernoulli=True, dataSet = 'training'):
	X, labels = load_mnist(dataSet, digits)
	# Bernoulli Values
	if Bernoulli:
		X = np.array(( np.random.rand(*X.shape) < X), dtype=float)
	else:
		X = np.array(X, dtype=float)
		
	total = len(X)
	size = (len(X[0])*len(X[0][0]))
	X = X.reshape(total,size)
	testSet = np.array(X, dtype=float)
	del X
	return testSet

def showASCIIImage(data):
	print data
	sys.exit()
	tmp = []
	for x in xrange(0,28):
		row = []
		for y in xrange(0,28):
			i = x*28+y
			row.append(data[i])
		tmp.append(row)
	#data = np.array(tmp, dtype=np.uint8)
	data = np.array(tmp, dtype=float)
	for d in data:
		row = ""
		for pixel in d:
			if pixel:
				row +="*"
			else:
				row+=" "
		print row

def saveImage(data, fileName='image.png'):
	try:
		tmp = []
		for x in xrange(0,28):
			row = []
			for y in xrange(0,28):
				i = x*28+y
				row.append((data[i], data[i], data[i]))
			tmp.append(row)
		data = np.array(tmp, dtype=np.uint8)
		print "Grabando imagen...",data.shape
		img = Image.fromarray(data)
		img.save(fileName)
	except Exception, e:
		print e

if __name__=='__main__':
	sys.exit(main(sys.argv))

