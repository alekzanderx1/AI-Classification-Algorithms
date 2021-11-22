import numpy as np


### Assignment 3 ###

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.X_train = X
		self.y_train = y
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		result = list()
		for x_test in X:
			# for each instance in test dataset, find distance to all instances in test dataset
			neighbourDistances = [(self.distance(x_test,x_train)) for x_train in self.X_train]

			# enumerating to keep track of indices
			enumNeighbours = enumerate(neighbourDistances)
			
			# sort the distances in ascending order and take the "K" closest ones
			kNeighbours = sorted(enumNeighbours, key=lambda x: x[1])[:self.k]

			# find the label of the nearest neighbours
			labels = [self.y_train[row[0]] for row in kNeighbours]

			# consider the most common label among the k nearest neighbours as the prediction
			prediction = max(set(labels), key=labels.count)

			result.append(prediction)

		return np.array(result)


class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		count = 0
		for i in range(len(X)):
			# stop training if number of steps are exhausted
			if count == steps:
				break
			count+=1

			# activation will contain positive or negative decimal number
			activation = np.dot(X[i],self.w) + self.b

			if activation > 0:
				prediction = 1
			else:
				prediction = 0

			# update weights if label is misclassified 
			if y[i] != prediction:
				self.w = self.w + ((self.lr)*X[i]*(prediction - y[i]))

		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		res = []
		for i in range(len(X)):
			activation = np.dot(X[i],self.w) + self.b[0]
			if activation > 0:
				prediction = 1
			else:
				prediction = 0
			res.append(prediction)
		return np.array(res)


class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def entropy(self, y_train):
		# find unique label and their counts
		values, counts = np.unique(y_train, return_counts=True)
		entropies = []
		for i in range(len(values)):
			probability = counts[i]/np.sum(counts)
			entropies.append(-probability*np.log2(probability))
		# calculate sum of individual entropy values
		return np.sum(entropies)

	def informationGain(self, X_train, y_train, featureIdx):
		# calculate total entropy of the current feature
		totalEntropy = self.entropy(y_train)

		#find unique feature values and their count and calculate weighted entropy
		values, counts = np.unique(X_train[:,featureIdx], return_counts=True)
		totalCount = np.sum(counts)

		informationAD = []
		for i in range(len(values)):
			value_probability = counts[i]/totalCount
			#calculating entropy using label values having index of examples with the current feature value
			value_entropy = self.entropy(y_train[np.where(X_train[:,featureIdx] == values[i])])
			informationAD.append(value_probability*value_entropy)
		totalIAD = np.sum(informationAD)

		# return Information Gain of the feature
		return totalEntropy - totalIAD

	# returns mode of the label values
	def plurality_value(self, y_train):
		return max(set(y_train), key=list(y_train).count)

	# contructs decision tree recursively using Training examples
	def decision_tree(self, X_train, y_train, features, parent_X_train=None, parent_y_train=None):
		labels = np.unique(y_train)
		# Base cases
		# If label is same for all examples then return the value of label
		if len(labels) == 1:
			return labels[0]
		# if examples are empty return plurality of parent examples
		elif len(X_train) == 0:
			return self.plurality_value(parent_y_train)
		# if all features are explored return plurality of the remaining examples
		elif len(features) == 0:
			return self.plurality_value(y_train)
		else:
			# find infromation gain of all features yet to be explored in this tree
			gain_values = []
			for featureIdx in features:
				gain_values.append((self.informationGain(X_train,y_train,featureIdx),featureIdx))

			# select feature with highest information gain
			gain_values.sort(key=lambda x:x[0])
			bestFeatureIdx = gain_values[-1][1]
			# contruct tree with root as the best feature
			tree = {bestFeatureIdx: {}}

			# remove best feature from list of features to explore next
			pruned_features = [feature for feature in features if feature != bestFeatureIdx]

			# contruct decision tree recursively for each value of the best feature
			parentUniqueValues = np.unique(X_train[:,bestFeatureIdx])
			for value in parentUniqueValues:
				# split example data to retain only the rows with given feature value
				splitIdx = np.where(X_train[:,bestFeatureIdx] == value)
				split_X_train = X_train[splitIdx]
				split_y_train = y_train[splitIdx]

				subtree = self.decision_tree(split_X_train,split_y_train, pruned_features,X_train,y_train)
				tree[bestFeatureIdx][value] = subtree

		return tree

	def makePrediction(self, x_train, tree):
		for featureIdx in range(x_train.shape[0]):
			if featureIdx in list(tree.keys()):
				try:
					result = tree[featureIdx][x_train[featureIdx]]
				except:
					# returning None as there is no node with given feature value in this subtree
					return None
				result = tree[featureIdx][x_train[featureIdx]]
				if isinstance(result,dict):
					return self.makePrediction(x_train, result)
				else:
					return result

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		features = [i for i in range(categorical_data.shape[1])]
		self.tree = self.decision_tree(categorical_data,y,features,None,None)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		predictions = []
		for x_train in categorical_data:
			predictions.append(self.makePrediction(x_train, self.tree))
		return np.array(predictions)


### Assignment 4 ###

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset

