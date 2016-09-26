import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy import linalg, sparse, misc
from sklearn.preprocessing import OneHotEncoder

# data loaded into X_bin_* and y_bin_*
data_train = pandas.read_csv("mnist_train.csv", engine='python', sep=',').as_matrix()
data_test = pandas.read_csv("mnist_test.csv", engine='python', sep=',').as_matrix()

data_train_bin = np.matrix([row for row in data_train if row[0] in [0, 1]])
x_bin_train = data_train_bin[:,1:]
y_bin_train = data_train_bin[:,0]

data_test_bin = np.matrix([row for row in data_test if row[0] in [0, 1]])
x_bin_test = data_test_bin[:,1:]
y_bin_test = data_test_bin[:,0]

def accuracy(y_test, y_predict):
	correct = 0
	for i in range(len(y_test)):
		if (y_test[i] == y_predict[i]):
			correct += 1
	return correct * 1. / len(y_test)

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def log_likelihood(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	mu = sigmoid(X * theta / (10 * len(X)))
	mu[~y_bool] = 1 - mu[~y_bool]	
	return (np.log(mu).sum() - reg * np.inner(theta, theta) / 2).item(0)

def grad_log_likelihood(X, y, theta, reg=1e-6):
	X = np.matrix(X)	
	return X.transpose() * (sigmoid(X * theta) - y) + reg * theta

def newton_log_likelihood(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	mu = sigmoid(X * theta)
	mu[~y_bool] = 1 - mu[~y_bool]	
	return (np.log(mu).sum() - reg * np.inner(theta, theta) / 2).item(0)

def newton_step(X, y, theta, reg=1e-6):
	X = np.matrix(X)
	mu = np.array([m.item(0) for m in sigmoid(X * theta)])
	return linalg.cho_solve(		
		linalg.cho_factor(X.transpose() * sparse.diags(mu * (1 - mu), 0) * X + reg * sparse.eye(X.shape[1])),
		grad_log_likelihood(X, y, theta, reg=reg),
	)

def linreg_grad(
	X, y,
	reg=1e-6, lr=1e-3, tol=1e-6,
	max_iters=500,
	print_freq=1
):
	y = y.astype(bool)
	y_bool = np.array([yy.item(0) for yy in y], dtype=bool)

	theta = np.matrix(np.zeros(X.shape[1])).transpose()

	objective = [log_likelihood(X, y_bool, theta, reg=reg)]
	grad = grad_log_likelihood(X, y, theta, reg=reg)

	while len(objective) - 1 <= max_iters and np.linalg.norm(grad) > tol:
		if (len(objective) - 1) % print_freq == 0:
			print('[i={}] likelihood: {}. grad norm: {}'.format(
				len(objective) - 1, objective[-1], np.linalg.norm(grad),
			))

		grad = grad_log_likelihood(X, y, theta, reg=reg)			
		theta = theta - lr * grad
		objective.append(log_likelihood(X, y_bool, theta, reg=reg))

	print('[i={}] done. grad norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(grad)
	))

	return theta, objective

def linreg_newton(
	X, y,
	reg=1e-6, tol=1e-6, max_iters=300,
	print_freq=5,
):
	y = y.astype(bool)
	y_bool = np.array([yy.item(0) for yy in y], dtype=bool)

	theta = np.matrix(np.zeros(X.shape[1])).transpose()
	
	objective = [newton_log_likelihood(X, y_bool, theta, reg=reg)]
	step = newton_step(X, y, theta, reg=reg)
	
	while len(objective)-1 <= max_iters and np.linalg.norm(step) > tol:
		if (len(objective)-1) % print_freq == 0:
			print('[i={}] likelihood: {}. step norm: {}'.format(
				len(objective)-1, objective[-1], np.linalg.norm(step)
			))

		step = newton_step(X, y, theta, reg=reg)
		theta -= step		
		objective.append(newton_log_likelihood(X, y_bool, theta, reg=reg))
	
	print('[i={}] done. step norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(step)
	))

	return theta, objective

def part_a_graph_1():
	theta_grad, objective_grad = linreg_grad(x_bin_train, y_bin_train, max_iters=500)
	theta_newton, objective_newton = linreg_newton(x_bin_train, y_bin_train, max_iters=300)

	plt.ylim([-8500, 500])
	plt.xlim([-20,520])
	plt.plot([x for x in range(len(objective_grad))], objective_grad, color='blue')
	plt.plot([x for x in range(len(objective_newton))], objective_newton, color='red')
	plt.ylabel('Iteration')
	plt.ylabel('Likelihood')
	plt.show()

# part_a_graph_1()

def get_accuracy_for_lambda(l):
	theta_newton, objective_newton = linreg_newton(x_bin_train, y_bin_train, 
		max_iters=300, reg=l)
	y_predicted = x_bin_test * theta_newton

	for i in range(len(y_predicted)):
		if y_predicted[i] > 0.:
			y_predicted[i] = 1
		else:
			y_predicted[i] = 0

	accu = accuracy(y_bin_test, y_predicted)
	print 'Newton accuracy: {}'.format(accu)
	return accu

def part_a_graph_2():
	ls = [1e-6] + [3.0 * i for i in range(1, 10)]
	plt.plot(ls, [get_accuracy_for_lambda(l) for l in ls], color='blue')	
	plt.ylim([0, 1.1])
	plt.ylabel('Accuracys')
	plt.ylabel('Lambda')
	plt.show()

# part_a_graph_2()

### Part b ###

def softmax(x):
	s = np.exp(x - np.max(x, axis = 1))
	return s / np.sum(s, axis=1)

def log_softmax(x):
	return x - misc.logsumexp(x, axis=1)

def softmax_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	W = np.matrix(W)
	W_Transpose = np.transpose(W)

	mu = X * W
	return np.sum(mu[y_one_hot] - misc.logsumexp(mu, axis =1)) - reg * np.einsum('ij,ji->', W_Transpose, W)/2

def soft_grad_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	print X.shape
	X_Transpose = np.transpose(X)
	W = np.matrix(W)
	print W.shape
	mu = X * W
	mu = np.exp(mu- np.max(mu, axis=1))
	mu = mu / np.sum(mu, axis=1)
	return X_Transpose * (mu-y_one_hot) + reg*W

def softmax_grad(
	X, y, reg=1e-6, lr=1e-8, tol=1e-6,
	max_iters=300, batch_size=256,
	verbose=False, print_freq=5):

	enc = OneHotEncoder()
	y_one_hot = enc.fit_transform(y.copy().reshape(-1,1)).astype(bool).toarray()
	W = np.zeros((X.shape[1], y_one_hot.shape[1]))
	ind = np.random.randint(0, X.shape[0], size=batch_size)
	objective = [softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)]
	grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)

	while len(objective)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if verbose and (len(objective)-1) % print_freq == 0: print('[i={}] likelihood: {}. grad norm: {}'.format(
			len(objective)-1, objective[-1], np.linalg.norm(grad)
			))

		ind = np.random.randint(0, X.shape[0], size=batch_size)
		grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)
		W = W - lr * grad

		objective.append(softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg))

	print('[i={}] done. grad norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(grad)
		))

	return W, objective

def part_b():
	w, objective = softmax_grad(x_bin_train, y_bin_train, max_iters=500)

	plt.ylim([-8500, 500])
	plt.xlim([-20,520])
	plt.plot([x for x in range(len(objective))], objective, color='blue')
	plt.xlabel('Iteration')
	plt.ylabel('Likelihood')
	plt.show()

# part_b()

### Part C ###

def predict_knn(X_test, X_train, y_train, k=5):
	num_data = X_test.shape[0]
	y_pred = [0] * num_data

	for i in range(num_data):
		digit = X_test[i]
		index = np.argpartition(1. / np.linalg.norm(X_train - digit[:,np.newaxis].T, axis=1), -k)[-k:]
		y_pred[i] = np.argmax(np.bincount(y_train[index]))

	return y_pred

def part_c():
	num = 2500
	X_train_sample = data_train[:num,1:]
	Y_train_sample = data_train[:num,0]
	X_test_sample = data_test[:num,1:]
	Y_test_sample = data_test[:num,0]

	for k in [1,5,10]:
		print('[k={}] accuracy: {}'.format(
			k,
			accuracy(Y_test_sample, predict_knn(X_test_sample, X_train_sample, Y_train_sample, k=k)),
		))

# part_c()