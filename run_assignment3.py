import assignment3 as models
import numpy as np
import sys

if(sys.version_info[0] < 3):
	raise Exception("This assignment must be completed using Python 3")

#==========================================================Data==========================================================
# Number of Instances:	
# 653
# Number of Attributes:
# 35 numeric, predictive attributes and the class

# Attribute Information:

# We have 35 variables for 653 counties, including demographics, covid info, previous election 
# results, work related information.
# percentage16_Donald_Trump	
# percentage16_Hillary_Clinton	
# total_votes20	
# latitude	
# longitude	
# Covid Cases/Pop	
# Covid Deads/Cases	
# TotalPop	
# Women/Men
# Hispanic
# White	
# Black	
# Native	
# Asian	
# Pacific	
# VotingAgeCitizen	
# Income	
# ChildPoverty	
# Professional	
# Service	
# Office	
# Construction	
# Production	
# Drive	
# Carpool	
# Transit	
# Walk	
# OtherTransp	
# WorkAtHome	
# MeanCommute	
# Employed	
# PrivateWork	
# SelfEmployed	
# FamilyWork	
# Unemployment


# Class Distribution:
# 328 - Candidate A (1), 325 - Candidate B (0)
#========================================================================================================================

def train_test_split(X, y, test_ratio):
	tr = int(y.size*test_ratio)
	return X[:tr], X[tr:], y[:tr], y[tr:]

def load_data(path):
	data = np.genfromtxt(path, delimiter=',', dtype=float)
	return data[:,:-1], data[:,-1].astype(int)

X, y = load_data("county_statistics.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.75)

#Initialization
#KNN
k = 3
knn = models.KNN(k)
#Perceptron
lr = .001
w = np.random.normal(0, .1, size=X_train.shape[1])	#Each column represents all the weights going into an output node
b = np.random.normal(0, .1, size=1)
perceptron = models.Perceptron(w, b, lr)
#MLP
lr = .0001
w1 = np.random.normal(0, .1, size=(X_train.shape[1], 10))
w2 = np.random.normal(0, .1, size=(10,1))
b1 = np.random.normal(0, .1, size=(1,10))
b2 = np.random.normal(0, .1, size=(1,1))
#mlp = models.MLP(w1, b1, w2, b2, lr)

#Train
steps = 100*y_train.size
knn.train(X_train, y_train)
perceptron.train(X_train, y_train, steps)
#mlp.train(X_train, y_train, steps)

#Check weights (For grading)
perceptron.w
perceptron.b
# mlp.w1
# mlp.b1
# mlp.w2
# mlp.b2

#Evaluate
def evaluate(solutions, real):
	if(solutions.shape != real.shape):
		raise ValueError("Output is wrong shape.")
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

solutions = knn.predict(X_test)
print(evaluate(solutions, y_test))
solutions = perceptron.predict(X_test)
print(evaluate(solutions, y_test))
#solutions = mlp.predict(X_test)
#print(evaluate(solutions, y_test))


#Initialization
#ID3
nbins = 3
data_range = (X_train.min(0), X_train.max(0))
id3 = models.ID3(nbins, data_range)

#Train
id3.train(X_train, y_train)

#Evaluate
def evaluate(solutions, real):
	if(solutions.shape != real.shape):
		raise ValueError("Output is wrong shape.")
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

solutions = id3.predict(X_test)
print(evaluate(solutions, y_test))

