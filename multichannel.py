import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import savefig
import numpy as np
from PIL import Image, ImageDraw
import math
import pylab
from time import time
from sklearn.cross_validation import train_test_split
from sklearn import metrics
channel = 1
size = 54

import csv

from scipy.misc import imread
import cv2


start = time()

entirePath = './data/train/entireImage/'
boudaryPath = './data/train/'
detPath = './data/train/c'

import pickle	
def save_paramsList(param_path,paramsList):		  
		write_file = open(param_path, 'wb')	 
		pickle.dump(paramsList,write_file,-1)
		write_file.close()

def load_paramsList(params_file):
	f=open(params_file,'rb')
	paramsList=pickle.load(f)
	f.close()
	return paramsList


############################################
#### load image and extract wells###########
############################################

def loadEntireImage(entire_path):
	
	#for other method
	#img = imread(entire_path,0)
	
	#for hoG method
	img = cv2.imread(entire_path,0)
	
	#print(img.shape)
	return img
	
def loadBoundary(ai):
	oneTest = ai
	x1 = 0
	y1 = 0
	x = round(float(oneTest[0]))
	y = round(float((oneTest[1])))
	w = round(float((oneTest[2])))
	h = round(float((oneTest[3])))
	
	if x + w <= 1080 and y + h <= 1280:
			x1 = x + 53
			y1 = y + 53# fix well size to get trainSetData balanced, size for an array in it 53*53 = 2809
	
	return([x,y,x1,y1])

def loadBoundaryW(path): 

	a = []
	csv_reader = csv.reader(open(path))
	for row in csv_reader:
		a.append(row)
	
	return a

def loadCoordinateW(path): #new

	a = []
	csv_reader = csv.reader(open(path))
	for row in csv_reader:
		a.append(row)
	
	arrayX = []
	arrayY = []
	
	for i in range(0,len(a)):
		xc = float(a[i][0]) #or int
		yc = float(a[i][1])
		
		arrayX.append(xc)
		arrayY.append(yc)
	
	
	return arrayX, arrayY

def loadCoordinate(path, bi): #bi for a well boundary in csv in string format
	a = []
	csv_reader = csv.reader(open(path))
	for row in csv_reader:
		a.append(row)
	
	arrayX = []
	arrayY = []
	
	x = float(bi[0])
	y = float(bi[1])
	w = float(bi[2])
	h = float(bi[3])

	
	for i in range(0,len(a)):
		xc = float(a[i][0])
		yc = float(a[i][1])
		if xc > x and xc < x+w: #need well validation check? or not?
			if yc > y and yc < y+h:
				arrayX.append(xc)
				arrayY.append(yc)
	return arrayX,arrayY
	
def printCoordinate(X,Y,img):
	draw = ImageDraw.Draw(img)
	for i in range(len(X)):
		draw.ellipse((X[i]-1,Y[i]-1,X[i]+1,Y[i]+1),fill = 'yellow', outline ='yellow')
	imgfig = plt.imshow(img)
	imgfig.set_cmap('hot')
	plt.axis('off')
	# wellfig = pylab.gcf()
	plt.savefig('./data/save/wellWithCoordinate.png',bbox_inches='tight')

	
	
############################################
######## generate training data ############
############################################


from numpy import random
from sklearn import preprocessing
from skimage.feature import hog
from skimage import data, color, exposure
from scipy import ndimage

def chooseRandomCrop(img,a,b):
	sizeX = img.size[0]
	sizeY = img.size[1]
	randomX = random.randint(0,sizeX-a)
	randomY = random.randint(0,sizeY-b)
	bound = [randomX,randomY,randomX+a,randomY+b]
	crop = img.crop(bound)
	return crop,bound


def countCell(cx): 
	return len(cx)

def generateTrainSetBalanced(entireImgPath,boundaryPath,coordinatePath,numberOfImage):
	
	 
	labelOneBoundList = []
	trainData = []
	trainLabelClassifier = []
	trainLabelCount=[]
	labelOneBoundList = []
	labelOneCountList = []
	
	labelOneNum=0
	labelZeroNum=0
	allBound = []
	
	
	for imageNumber in range(numberOfImage[0],numberOfImage[1]+1):
		'''
		#Previous figure
		entireImg0 = loadEntireImage(entireImgPath+str(imageNumber)+"00.tif")
		#Later figure
		if imageNumber == 4 or imageNumber == 8 or imageNumber == 12:
			entireImg1 = loadEntireImage(entireImgPath+str(imageNumber)+"01.jpeg")
			b, g, r = cv2.split(entireImg1)
			entireImg1 = b
		else:
			entireImg1 = loadEntireImage(entireImgPath+str(imageNumber)+"02.tif")
		'''
		# Original figure
		entireImg2 = loadEntireImage(entireImgPath+str(imageNumber)+"01.jpeg")
		
		#b, g, r = cv2.split(entireImg2)
		'''
		# Fluorescent figure
		entireImg0 = loadEntireImage(entireImgPath+str(imageNumber)+"03.tif")
		entireImg1 = loadEntireImage(entireImgPath+str(imageNumber)+"04.tif")
		
		b = preprocessing.scale(b)
		#merged = [b]
		
		
		#print(entireImg0.shape)
		
		#proprocess all data to (-1,1)	 -----0.95
		entireImg0 = preprocessing.scale(entireImg0)
		entireImg1 = preprocessing.scale(entireImg1)
		#entireImg2 = preprocessing.scale(entireImg2)
		#entireImg3 = preprocessing.scale(entireImg3)
		#entireImg4 = preprocessing.scale(entireImg4)
 
		merged = [entireImg0, entireImg1, b] 
		
		
		#sobel part	  -------0.95
		entireImg2 = color.rgb2gray(entireImg2)
		sx2 = ndimage.sobel(entireImg2, axis=0, mode='constant')
		sy2 = ndimage.sobel(entireImg2, axis=1, mode='constant')
		#sx2 = preprocessing.scale(sx2)
		
		merged = [sx2,entireImg2, sy2]
	    
	
		merged = np.array(merged)
		'''
		
		boundaryW = loadBoundaryW(boundaryPath+str(imageNumber)+".csv")
		
		for i in range(len(boundaryW)): #for a single well
			boundary = loadBoundary(boundaryW[i])
			cxx, cyy = loadCoordinate(coordinatePath+str(imageNumber)+".csv", boundaryW[i])
			#well = merged[:,boundary[1]:boundary[3],boundary[0]:boundary[2]] 
			
			well = entireImg2[boundary[1]:boundary[3],boundary[0]:boundary[2]] 
			#print(well2.shape)
			x,y = well.shape
			if x!=53 or y!=53:
				continue
			#well = well.reshape(channel,z*y)
			
			
			#hog part
			
			hist2 = cv2.calcHist([well],[0],None,[107],[0,255])
			well = well.reshape(53*53,1)
			'''
			hist2 = hist2.reshape(15,15)
			new_im = Image.fromarray(hist2.astype(np.uint8))
			hist = new_im.resize((size,size))
			hist = np.array(hist)
			
			new_im2 = Image.fromarray(well2.astype(np.uint8))
			well2 = new_im2.resize((size,size))
			well2 = np.array(well2)			
			'''
			

			well = np.vstack((well,hist2))
			
			
			well = [well]
			well = np.array(well)
			#print(hist.shape)
			well = well.reshape(channel,size*size)
			
			
			count = countCell(cxx)
			if(count>0 and boundary[2] != 0):
				
				labelOneNum+=1
				label = 1			  
				trainLabelClassifier.append(label)
				trainData.append(well)
				trainLabelCount.append(count)
				allBound.append(boundary)
				
				labelOneCountList.append(count)
				labelOneBoundList.append(boundary)
				
			elif(count==0 and boundary[2] != 0):
				labelZeroNum+=1
				label=0
				trainLabelClassifier.append(label)
				trainData.append(well)
				trainLabelCount.append(count)	
				allBound.append(boundary)
			
	print("Generate balanced train set.")
	print(str(len(labelOneCountList))+" in "+str(len(allBound))+" has label.") 
	return trainData,trainLabelClassifier,trainLabelCount,labelOneCountList,labelOneBoundList,allBound
	
	
	
print('finish')


	
trainSetData,trainSetLabel,trainSetCountLabel,labelOneCount,labelOneBound,allBound = generateTrainSetBalanced(entirePath,boudaryPath,detPath,[1,12])

Train0 = []
Label0 = []

Train1 = []
Label1 = []

Train2 = []
Label2 = []

Train3 = []
Label3 = []

Train4 = []
Label4 = []

Train5andMore = []
Label5andMore = []

for i in range(len(trainSetData)):
	if trainSetCountLabel[i] == 0:
		Train0.append(trainSetData[i])
		Label0.append(trainSetCountLabel[i])
	if trainSetCountLabel[i] == 1:
		Train1.append(trainSetData[i])
		Label1.append(trainSetCountLabel[i])
	if trainSetCountLabel[i] == 2:
		Train2.append(trainSetData[i])
		Label2.append(trainSetCountLabel[i])		
	if trainSetCountLabel[i] == 3:
		Train3.append(trainSetData[i])
		Label3.append(trainSetCountLabel[i])
	if trainSetCountLabel[i] == 4:
		Train4.append(trainSetData[i])
		Label4.append(trainSetCountLabel[i])
	if trainSetCountLabel[i] >= 5:
		Train5andMore.append(trainSetData[i])
		Label5andMore.append(trainSetCountLabel[i])
		
		
newTrainSetData = []
newTrainSetCountLabel = []

newTestSetData = []
newTestSetCountLabel = []


#import random
#order0 = random.sample(range(len(Train0)),85)
order0 = [667, 451, 58, 207, 9, 746, 86, 490, 1092, 54, 594, 804, 241, 1070, 1053, 436, 688, 582, 507, 144, 1074, 
			381, 546, 1060, 777, 863, 564, 1177, 395, 846, 1046, 1155, 388, 793, 1087, 1015, 157, 837, 444, 160, 
			809, 159, 469, 649, 294, 155, 1110, 28, 356, 1062, 41, 352, 7, 550, 871, 379, 387, 47, 606, 233, 592, 
			82, 240, 601, 148, 1101, 655, 79, 492, 1171, 4, 738, 984, 146, 693, 478, 847, 1038, 1124, 853, 563, 140, 705, 765, 427]
order1 = [148, 199, 120, 179, 215, 59, 219, 110, 58, 162, 92, 102, 231, 8, 16, 156, 11, 145, 166, 210, 222, 220, 193, 
			12, 211, 105, 146, 10, 126, 177, 94, 217, 54, 70, 104, 133, 3, 213, 138, 7, 163, 204, 67, 212, 229, 129, 
			1, 108, 150, 230, 28, 60, 144, 32, 13, 137, 132, 45, 21, 48, 206, 23, 122, 164, 196, 71, 38, 81, 64, 97, 
			128, 192, 47, 119, 100, 111, 109, 17, 197, 18, 173, 165, 115, 77, 176]
#random.sample(range(len(Train1)),85)
order2 = [76, 136, 35, 263, 311, 328, 69, 95, 11, 21, 63, 437, 179, 138, 46, 414, 31, 130, 288, 241, 223, 298, 395, 
			355, 472, 474, 15, 341, 403, 228, 147, 426, 205, 408, 416, 312, 322, 244, 163, 377, 218, 465, 119, 453, 
			190, 307, 415, 85, 90, 158, 373, 127, 455, 171, 464, 279, 267, 162, 229, 314, 245, 247, 338, 459, 66, 313, 
			434, 172, 6, 379, 387, 475, 234, 93, 407, 189, 59, 14, 354, 441, 71, 58, 384, 275, 116]
#random.sample(range(len(Train2)),85)
order3 = [89, 88, 98, 92, 2, 78, 128, 20, 12, 132, 80, 123, 62, 26, 57, 10, 41, 0, 39, 71, 9, 121, 100, 69, 60, 15, 
			35, 99, 76, 86, 74, 112, 66, 105, 108, 4, 46, 63, 45, 37, 114, 64, 50, 115, 83, 24, 129, 104, 11, 43, 127, 
			52, 107, 117, 19, 6, 17, 68, 32, 134, 116, 13, 81, 72, 113, 82, 84, 106, 77, 29, 67, 91, 21, 96, 101, 22, 
			131, 87, 119, 94, 44, 42, 33, 48, 90]
#random.sample(range(len(Train3)),85)
order4 = [91, 149, 119, 102, 68, 92, 54, 123, 141, 174, 35, 74, 40, 155, 125, 118, 93, 169, 5, 115, 52, 161, 17, 188,
			73, 20, 36, 145, 182, 100, 69, 168, 192, 78, 84, 112, 104, 23, 186, 160, 7, 87, 25, 103, 122, 18, 76, 147,
			143, 108, 157, 21, 177, 154, 46, 33, 37, 43, 81, 31, 133, 27, 30, 171, 184, 116, 83, 127, 32, 28, 88, 94, 
			44, 63, 151, 138, 144, 55, 180, 65, 79, 140, 71, 19, 96]
#random.sample(range(len(Train4)),85)
order5 = [64, 1, 136, 155, 108, 191, 156, 63, 95, 22, 97, 78, 131, 114, 171, 115, 190, 25, 72, 80, 180, 168, 159, 162,
			215, 6, 98, 206, 164, 147, 127, 14, 200, 173, 0, 197, 172, 62, 94, 85, 15, 76, 36, 167, 101, 142, 61, 9, 
			27, 184, 170, 135, 37, 83, 212, 119, 29, 73, 54, 117, 32, 145, 189, 202, 57, 5, 152, 49, 208, 100, 71, 50,
			59, 8, 74, 151, 18, 187, 198, 149, 65, 103, 204, 201, 93]
#random.sample(range(len(Train5andMore)),85)


for i in range(85):
	if i < 2:
		newTrainSetData.append(Train0[order0[i]])
		newTrainSetCountLabel.append(Label0[order0[i]])
		
		newTrainSetData.append(Train1[order1[i]])
		newTrainSetCountLabel.append(Label1[order1[i]])
		
		newTrainSetData.append(Train2[order2[i]])
		newTrainSetCountLabel.append(Label2[order2[i]])
		
		newTrainSetData.append(Train3[order3[i]])
		newTrainSetCountLabel.append(Label3[order3[i]])
		
		newTrainSetData.append(Train4[order4[i]])
		newTrainSetCountLabel.append(Label4[order4[i]])
		
		newTrainSetData.append(Train5andMore[order5[i]])
		newTrainSetCountLabel.append(Label5andMore[order5[i]])
		
	if i>=70 :
		newTestSetData.append(Train0[order0[i]])
		newTestSetCountLabel.append(Label0[order0[i]])
		
		newTestSetData.append(Train1[order1[i]])
		newTestSetCountLabel.append(Label1[order1[i]])
		
		newTestSetData.append(Train2[order2[i]])
		newTestSetCountLabel.append(Label2[order2[i]])
		
		newTestSetData.append(Train3[order3[i]])
		newTestSetCountLabel.append(Label3[order3[i]])
		
		newTestSetData.append(Train4[order4[i]])
		newTestSetCountLabel.append(Label4[order4[i]])
		
		newTestSetData.append(Train5andMore[order5[i]])
		newTestSetCountLabel.append(Label5andMore[order5[i]])		
		

		
print(len(newTrainSetData))
print(len(newTestSetData))

'''
out = random.sample(range(len(newTrainSetData)),20)
for i in out:
	img = newTrainSetData[i]
	img = img.reshape(53,53)
	new_im = Image.fromarray(img)
	new_im.save('./'+str(i)+'.jpg')
'''

trainList = [newTrainSetData,newTrainSetCountLabel]
save_paramsList('./params/trainList.pkl',trainList)
testList = [newTestSetData,newTestSetCountLabel]
save_paramsList('./params/testList.pkl',testList)


############################################
######## build the CNN model ###############
############################################

import os
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from sklearn.metrics import mean_squared_error


from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from pydoc import help
from scipy.stats.stats import pearsonr
from math import sqrt

# draw the output figure of CNN	   
def ScatterFig(predict,true,number,title):
	pylab.rcParams['figure.figsize'] = (10.0,10.0)
	img = plt.scatter(true,predict,	 color='black')	   
	plt.ylabel('Predicted Number Of Cells')
	plt.xlabel('True Number Of Cells')
	plt.title(title)
	plt.axis('on')
	fig = plt.gcf()

	fig.savefig('./resultFig/image'+str(number)+'.png',dpi=100)
	plt.show(img)
	print('MSE:')
	print(mean_squared_error(predict,true))
	print('RMSE:')
	print(sqrt(mean_squared_error(predict,true)))
	
	print('PEARSON:')
	print(pearsonr(true,predict))
	


start = time()

def load_data(Data,Label):
	train_data_list,test_data_valid, train_label_list, test_label_valid = train_test_split(Data,Label, test_size=0.2, random_state=0)
	test_data_list,valid_data_list,test_label_list,valid_label_list= train_test_split(test_data_valid,test_label_valid,test_size=0.5,random_state=0)
	
	train_data = numpy.array(train_data_list)
	train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
	train_label = numpy.array(train_label_list)
	
	valid_data = numpy.array(valid_data_list)
	valid_data = valid_data.reshape(valid_data.shape[0],valid_data.shape[1]*valid_data.shape[2])
	valid_label = numpy.array(valid_label_list)
	
	test_data = numpy.array(test_data_list)
	test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
	test_label = numpy.array(test_label_list)
	
	# Data stored as shared type so that they can be copied to GPU and get speed increased.
	def shared_dataset(data_x, data_y, borrow=True):
		shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX)
								 ,borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX)
								,borrow=borrow)
		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_data,train_label)
	test_set_x, test_set_y = shared_dataset(test_data,test_label)
	valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
	
	rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)]
	return rval



#Classifier, the last layer of CNN, softmax used
class LogisticRegression(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
			value=numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)
		self.b = theano.shared(
			value=numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


#Classifier, the last layer of CNN, softmax used
class LinearRegression(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
			value=numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)

		self.b = theano.shared(
			value=numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)

		self.p_y_given_x = T.dot(input, self.W) + self.b
#		  self.y_pred = T.argmax(self.p_y_given_x, axis=1)
#		  self.y_pred=T.dot(input, self.W) + self.b
		self.y_pred = self.p_y_given_x[:,0]
		self.params = [self.W, self.b]	   

	def linear_likelihood(self, y,number):
#		  return T.square(y-self.y_pred)
		return T.sum(T.pow(self.y_pred-y,2))/(2*number)
		return T.sum(T.pow(self.y_pred-y,2))

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
#			  return T.mean(T.neq(self.y_pred, y))
#			  return T.sum(T.sqr(y-self.y_pred),axis=1)
#			  return T.sum(T.sqr(y-self.y_pred))
#			  return T.sum(T.pow(self.y_pred-y,2))
			return T.sum(T.pow(self.y_pred-y,2))
#			  return mean_squared_error(self.y_pred, y)

		else:
			raise NotImplementedError()

#Full-connected layer, the layer before classifier
class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.tanh):

		self.input = input

		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_out)),
					high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]


#Convolutional layer + max-pooling layer
class LeNetConvPoolLayer(object):

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

		assert image_shape[1] == filter_shape[1]
		self.input = input

		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))

		# initialize weights with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# Convolutional
		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		# Max-pooling
		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]


#Save the parameters from training
def save_params(param_path,param1,param2,param3,param4):  
		import pickle  
		write_file = open(param_path, 'wb')	  
		pickle.dump(param1, write_file, -1)
		pickle.dump(param2, write_file, -1)
		pickle.dump(param3, write_file, -1)
		pickle.dump(param4, write_file, -1)
		write_file.close()	

		
print('finish')


############################################
######## begin train the CNN model #########
############################################

def train_CNN(width,height,patience,choice,number,cellNumber,learningRate):	 
  
	fullyOutputNumber = 1000
	n_epochs=500
	# nkerns: number of kernels in each layer
	nkerns=[5,10]
#settinghere

	layer1_conv = 5
	layer2_conv = 5
	
	batch_size = 1
	 
	#Initial parameter
	rng = numpy.random.RandomState(23455)
	#Load data
	if(choice=='logistic_zeroOne'):
		datasets = load_data(trainSetData,trainSetLabel)
		learning_rate=learningRate
	elif(choice=='logistic_count'):
		datasets = load_data(trainSetData,trainSetCountLabel)
		learning_rate=learningRate
	elif(choice=='linear_count'):
		datasets = load_data(trainSetData,trainSetCountLabel)
		learning_rate=learningRate


	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]


	#Calculate batch_size for each data set
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	print(n_train_batches)
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
   
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size

	#Define several variables, x as train data, as the input of layer0
	index = T.lscalar()
	x = T.matrix('x')
	#x = theano.config.floatX.xmatrix(theano.config.floatX)
	y = T.ivector('y')



	######################
	#Build CNN Model:
	#input+layer0(LeNetConvPoolLayer)+layer1(LeNetConvPoolLayer)+layer2(HiddenLayer)+layer3(LogisticRegression)
	######################
	print ('...building')



	# Reshape matrix of rasterized images of shape (batch_size, 50*50)
	# to a 4D tensor, compatible with our LeNetConvPoolLayer
	# (50,50) is the size of  images.
	layer0_input = x.reshape((batch_size, channel, width, height))

	# The first convolutional_maxpooling layer
	# Size after convolutional: (50-5+1 , 50-5+1) = (46, 46)
	# Size after maxpooling: (46/2, 46/2) = (23, 23), ignore the boundary
	# 4D output tensor is thus of shape (batch_size, nkerns[0], 23, 23)
	layer0 = LeNetConvPoolLayer(
		rng,
		input=layer0_input,
		image_shape=(batch_size, channel, width, height),
		filter_shape=(nkerns[0], channel, layer1_conv, layer1_conv),
		poolsize=(2, 2)
	)
   

	# Second convolutional + maxpooling layer, use last layer's output as input, (batch_size, nkerns[0], 23, 23)
	# 
	# Size after convolutional: (23-5+1 , 23-5+1) = (19, 19)
	# Size after maxpooling: (19/2, 19/2) = (9,9), ignore the boundary
	#todo: /2 problem
	# 4D output tensor is thus of shape (batch_size, nkerns[1], 9,9)
	
	
# [1]	
	width1 = math.floor((width-layer1_conv+1)/2)
	height1 = width1
	
	layer1 = LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		image_shape=(batch_size, nkerns[0], width1,height1),
		filter_shape=(nkerns[1], nkerns[0], layer2_conv,layer2_conv),
		poolsize=(2, 2)
	)
	
	hiddenlayerSize = math.floor((width1-layer2_conv+1)/2)
	
	



	#HiddenLayer full-connected layer, the size of input is (batch_size,num_pixels), so each sample will get a one-dimentional vector after layer0 and layer1
	#Output from last layer (batch_size, nkerns[1], 9,9) can be turned to (batch_size,nkerns[1]*9*9), by flatten

# [2]  
	layer2_input = layer1.output.flatten(2)
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		
#[2]
		n_in=nkerns[1] * hiddenlayerSize*hiddenlayerSize,
		n_out=fullyOutputNumber,	  #output number of full-connected layer, defined, can change
		activation=T.tanh
	)

 
	#Classifier Layer
	###############
	# Define some basic factors in optimization, cost function, train, validation, test model, updating rules(Gradient Descent)
	###############
	# Cost Function
 
	if(choice=='logistic_zeroOne'):
		layer3 = LogisticRegression(input=layer2.output, n_in=fullyOutputNumber, n_out=2)
		cost = layer3.negative_log_likelihood(y)
	elif(choice=='logistic_count'):
		layer3 = LogisticRegression(input=layer2.output, n_in=fullyOutputNumber, n_out=cellNumber+1) 
		cost = layer3.negative_log_likelihood(y)
	elif(choice=='linear_count'):
		layer3 = LinearRegression(input=layer2.output, n_in=fullyOutputNumber, n_out=1)	 
		cost = layer3.errors(y)
		#		  cost = layer3.linear_likelihood(y,number)
	
	
	test_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	# All parameters
	# [3]
	params = layer3.params + layer2.params + layer1.params + layer0.params
#	  params = layer3.params + layer2.params+layer4.params + layer1.params + layer0.params



	# Gradient of each parameter
	grads = T.grad(cost, params)
	# Updating rules
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	
	train_model = theano.function(
	[index],
		#[4]
	[cost,layer3.p_y_given_x, layer3.W, layer3.b, layer3.y_pred, layer2_input, layer2.output, y],
	updates=updates,
	givens={
		x: train_set_x[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
	)
	
	


	###############
	# Train CNN to find the best parameter
	###############
	print ('...training')
	patience_increase = 2  
	improvement_threshold = 0.99  
	print(n_train_batches)
	print(patience)
	validation_frequency = min(n_train_batches, patience / 2) 

	#print(validation_frequency)
	
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in range(math.floor(n_train_batches)):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print ('training @ iter = ', iter)
			cost_ij = train_model(minibatch_index)
			
			if (iter + 1) % validation_frequency == 0:
				
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in range(math.floor(n_valid_batches))]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					  (epoch, minibatch_index + 1, n_train_batches,
					   this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter
					# test it on the test set
					test_losses = [
						test_model(i)
						for i in range(math.floor(n_test_batches))
					]
					test_score = numpy.mean(test_losses)
					print(('	 epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))
		   
			if(choice=='logistic_zeroOne'):
				save_params('./params/logistic_zeroOne_params.pkl',layer0.params,layer1.params,layer2.params,layer3.params)#save parameter
			elif(choice=='logistic_count'):
				save_params('./params/logistic_count_params.pkl',layer0.params,layer1.params,layer2.params,layer3.params)#save parameter
			elif(choice=='linear_count'):
				#[5]
				save_params('./params/linear_count_params.pkl',layer0.params,layer1.params,layer2.params,layer3.params)#save parameter
#				  save_params('./params/linear_count_params.pkl',layer0.params,layer1.params,layer4.params,layer2.params,layer3.params)#save parameter



			if patience <= iter:
				done_looping = True
				break

	
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
   
   
[trainSetData,trainSetCountLabel] = load_paramsList('./params/trainList.pkl')
[testSetData,testSetCountLabel] = load_paramsList('./params/testList.pkl')

start = time()

train_CNN(size,size,5000,'linear_count',1000,16,0.00001) #input size 53
print()
print('Finish training in {:.2f} seconds.'.format(time()-start))



############################################
######## using the CNN model ###############
############################################
import sys
import pickle

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import math

# [1]
def load_params(params_file):
	f=open(params_file,'rb')
	layer0_params=pickle.load(f)
	layer1_params=pickle.load(f)
	layer2_params=pickle.load(f)
	layer3_params=pickle.load(f)
	f.close()
	return layer0_params,layer1_params,layer2_params,layer3_params

def load_data(Data,Label):

	#test_data = numpy.array(Data)
	#print(type(test_data))
	test_data = numpy.array(Data)
	test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
	test_label = numpy.array(Label)
	return test_data,test_label


class LogisticRegression(object):
	def __init__(self, input, params_W,params_b,n_in, n_out):
		self.W = params_W
		self.b = params_b
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()
			
			
			
class LinearRegression(object):
	def __init__(self, input, params_W,params_b,n_in, n_out):
		self.W = params_W
		self.b = params_b
		self.p_y_given_x = T.dot(input, self.W) + self.b
		self.y_pred = self.p_y_given_x[:,0]
#		  self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def linear_likelihood(self, y,number):
#		  return T.square(y-self.y_pred)
		return T.sum(T.pow(self.y_pred-y,2))/(2*number)

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.sum(T.pow(self.y_pred-y,2))
#			  return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


class HiddenLayer(object):
	def __init__(self, input, params_W,params_b, n_in, n_out,
				 activation=T.tanh):
		self.input = input
		self.W = params_W
		self.b = params_b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		self.params = [self.W, self.b]

	
class LeNetConvPoolLayer(object):
	def __init__(self,	input,params_W,params_b, filter_shape, image_shape, poolsize=(2, 2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input
		self.W = params_W
		self.b = params_b
	  
		conv_out = conv.conv2d(
		#conv_out = Conv3D.conv3D(	 
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)
	   
		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]

		
import matplotlib.pyplot as plt

def use_CNN(dataSet,labelSet,params_file,choice,printSwitch):	


#	  nkerns=[10, 20]
	data,label = load_data(dataSet,labelSet)
	# faces,label=load_data(dataset)
	#data = dataSet
	#label = labelSet
	data_num = len(data)   #how many data
  
	#load parameter
	#[2]
	layer0_params,layer1_params,layer2_params,layer3_params=load_params(params_file)
#	  layer0_params,layer1_params,layer4_params,layer2_params,layer3_params=load_params(params_file)



	
	x = T.matrix('x')  

	
	width = size
	height =size
	#settinghere
	nkerns=[5, 10]
	fullyOutputNumber = 1000

	layer1_conv = 5
	layer2_conv = 5
	
	layer0_input = x.reshape((data_num, channel, width,height)) 
	layer0 = LeNetConvPoolLayer(
		input=layer0_input,
		params_W=layer0_params[0],
		params_b=layer0_params[1],
		image_shape=(data_num, channel, width,height),
		filter_shape=(nkerns[0], channel, layer1_conv,layer1_conv),
		poolsize=(2, 2)
	)

	width1 = math.floor((width-layer1_conv+1)/2)
	
	#[3]
	layer1 = LeNetConvPoolLayer(
		input=layer0.output,
		params_W=layer1_params[0],
		params_b=layer1_params[1],
		image_shape=(data_num, nkerns[0], width1,width1),
		filter_shape=(nkerns[1], nkerns[0], layer2_conv, layer2_conv),
		poolsize=(2, 2)
	)
	
	layer2_input = layer1.output.flatten(2)

	width2 = math.floor((width1-layer2_conv+1)/2)
		
	layer2 = HiddenLayer(
		input=layer2_input,
		params_W=layer2_params[0],
		params_b=layer2_params[1],
		#[3]
		n_in=nkerns[1] * width2 * width2,
		n_out= fullyOutputNumber,	   
		activation=T.tanh
	)

	if(choice=='logistic_zeroOne'):
		layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in= fullyOutputNumber, n_out=2)
	elif(choice=='logistic_count'):
		layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=fullyOutputNumber, n_out=16)
	elif(choice=="linear_count"):
		layer3 = LinearRegression(input=layer2.output,params_W=layer3_params[0],params_b=layer3_params[1],n_in=fullyOutputNumber, n_out=1)	

	
	 
	 
	f = theano.function(
		[x],	
		layer3.y_pred
	)

	
	pred = f(data)

	

	wrongList = []
	rightList = []
#	  plt.plot(pred,label)

	for i in range(data_num):
		if label[i]==round(pred[i]) :
			rightList.append(i)
		try:
			if(label[i]!=round(pred[i]) or pred[i]==numpy.nan):
				wrongList.append(i)
				if(printSwitch==True):
					print('picture: %i is %i, mis-predicted as	%i' %(i, label[i], pred[i]))

		except:
			wrongList.append(i)
	print(str(len(wrongList))+" in "+str(len(label))+" has been predicted wrong")
	print("predict accuracy: {:.2f}%".format((len(label)-len(wrongList))/len(label)*100))
	return label, pred, wrongList,rightList
		
	
print('finish')


from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from pydoc import help
from scipy.stats.stats import pearsonr
[testSetData,testSetCountLabel] = load_paramsList('./params/testList.pkl')
y_true,y_pred,wrongID,rightID = use_CNN(testSetData, testSetCountLabel,'./params/linear_count_params.pkl','linear_count',False)

ScatterFig(y_pred,y_true,3,'Model with CNNs')





	
