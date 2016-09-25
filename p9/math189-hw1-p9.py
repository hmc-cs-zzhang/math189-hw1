import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# Read data from the iris data set
Data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv',  sep=',', engine='python')

X = Data[['Sepal.Length', 'Petal.Width']].as_matrix()
y = Data.Species.as_matrix()
n = len(X)
labels = np.unique(y)


# Create a list of y_index, in which 0 represents setosa, 1 represents
# versicolor and 2 represent virginica. Implement because please np.argmax
# return the indices on the prob with max indices 0, 1 or 2. When we compare results, we
# cannot directly compare y with np.argmax. We need to compare with 
# y-index, in which index matches the name of species in y.
y_index = []
for index in range(len(y)):
	if y[index] == 'setosa':
		y_index += [0]
	elif y[index] == 'versicolor':
		y_index += [1]
	else:
		y_index += [2]


def discriminant_analysis(X, y, linear = False, reg = 0.0):
	labels = np.unique(y)
	mu = {}
	cov = {}
	pi = {}
	for label in labels:
		pi[label] = (y == label).mean()
		mu[label] = X[y == label].mean(axis = 0)
		diff = X[y == label] - mu[label]
		diff_matrix = np.matrix(diff)		
		cov[label] = np.transpose(diff_matrix) * diff_matrix /(y == label).sum()

	# if the analysis is linear, we will transfor the covariance to a
	# single matrix for all species instread of dictionary
	if linear:
		cov = sum((y == label).sum() * cov[label] for label in labels)
		cov = cov / y.shape[0]
		cov = reg * np.diag(np.diag(cov)) + (1 - reg) * cov

	return pi, mu, cov


def normal_density(X, mu, cov):
	diff = X - mu
	diff_matrix = np.matrix(diff)	
	cov_inv = np.linalg.inv(cov)
	cov_det = np.linalg.det(cov)
	cov_det_sqrt = math.sqrt(cov_det)
	normal_Density = np.exp(-diff_matrix * cov_inv * np.transpose(diff_matrix) / 2) / ((2 * math.pi)**(n/2) * cov_det_sqrt)

	return normal_Density


# Calculate the Probablity of P(y|x) given p(y) and p(x|y) follow Bay's Theorem
def predict_proba(X, py, mu, cov):
	prob = np.zeros((X.shape[0], len(py)))
	
	if type(cov) is not dict:
		covariance = cov
		cov = defaultdict(lambda: covariance)

	# for each data point, calculate the probablity this data of being each 
	# of the species 
	for i, x in enumerate(X):
		for j in range(len(py)):
			# Use label because key of the dictionary is name of species
			label = labels[j]
			prob[i,j] = py[label] * normal_density(x, mu[label], cov[label])
	
	# sum prob of each data point for all three species and devided each
	# prob by this sum
	prob = prob / prob.sum(axis=1)[:,np.newaxis]

	return prob


# Use linespace to generate 50 different regularization parameter lambda
# Report Accuracy
for lambda_ in np.linspace(0.0,1.0,50):
    py, mu, cov = discriminant_analysis(X, y, linear = True, reg=lambda_)
    print('[linear=True,reg={:0.2f}] accuracy={:0.4f}'.format(lambda_, (np.argmax(predict_proba(X, py, mu, cov), axis=1) == y_index).mean()))


# Non-linear Gaussian Discriminant Analysis
py, mu, cov = discriminant_analysis( X, y,linear = False)

# Report Accuracy
print('[linear=False, reg=0.00] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, py, mu, cov), axis=1) == y_index).mean()))


# Plot
# Separate all data points
Setosa = Data[Data.Species == 'setosa']
Versicolor = Data[Data.Species == 'versicolor']
Virginica = Data[Data.Species == 'virginica']

Setosa_sepalL = Setosa['Sepal.Length']
Setosa_petalW = Setosa['Petal.Width']
Versicolor_sepalL = Versicolor['Sepal.Length']
Versicolor_petalW = Versicolor['Petal.Width']
Virginica_sepalL = Virginica['Sepal.Length']
Virginica_petalW = Virginica['Petal.Width']

mu_setosa_x = mu['setosa'][0]
mu_setosa_y = mu['setosa'][1]
mu_versicolor_x = mu['versicolor'][0]
mu_versicolor_y = mu['versicolor'][1]
mu_virginica_x = mu['virginica'][0]
mu_virginica_y = mu['virginica'][1]

plt.ylim([-0.1,3])
plt.xlim([4,8])
# Plot all data point
plt.plot(Setosa_sepalL, Setosa_petalW, 'ro', color='red')
plt.plot(Versicolor_sepalL, Versicolor_petalW, 'ro', color='blue')
plt.plot(Virginica_sepalL, Virginica_petalW, 'ro', color='green')
#Plot all mu coming from non-linear gaussian discriminant analysis
plt.plot([mu_versicolor_x],[mu_versicolor_y], 'ro', color='black')
plt.plot([mu_setosa_x],[mu_setosa_y], 'ro', color='black')
plt.plot([mu_virginica_x],[mu_virginica_y], 'ro', color='black')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.title('Gaussian Discriminant Analysis Non-linear')
plt.show()


