import numpy
import pandas
import random
import matplotlib.pyplot as plt

data = pandas.read_csv("online_news_popularity.csv", engine='python', sep=', ');

data['type'] = ''

data.loc[:int(2.0 / 3 * len(data)), 'type'] = 'train'
data.loc[int(2.0 / 3 * len(data)):int(5.0 / 6 * len(data)), 'type'] = 'validation'
data.loc[int(5.0 / 6 * len(data)):, 'type'] = 'test'

data.describe()

train_x = data[data.type == 'train'][[col for col in data.columns if col not in ['url', 'shares', 'type']]]
train_y = numpy.log(data[data.type == 'train'].shares).reshape(-1,1)

validation_x = data[data.type == 'validation'][[col for col in data.columns if col not in ['url', 'shares', 'type']]]
validation_y = numpy.log(data[data.type == 'validation'].shares).reshape(-1,1)

test_x = data[data.type == 'test'][[col for col in data.columns if col not in ['url', 'shares', 'type']]]
test_y = numpy.log(data[data.type == 'test'].shares).reshape(-1,1)

train_x = numpy.hstack((numpy.ones_like(train_y), train_x))
validation_x = numpy.hstack((numpy.ones_like(validation_y), validation_x))
test_x = numpy.hstack((numpy.ones_like(test_y), test_x))

def linreg(x, y, reg=0.0):
	eye = numpy.eye(x.shape[1])
	eye[0,0] = 0.
	x = numpy.matrix(x)
	y = numpy.matrix(y)
	xt = x.transpose()
	return numpy.linalg.solve(xt * x + reg * eye, xt * y)

def mse(wt):
	wt_matrix = numpy.matrix(wt)
	error = 0.0
	for i in range(len(validation_x)):
		val_x_matrix = numpy.matrix(validation_x[i])
		y_reg = val_x_matrix * wt_matrix
		error += (validation_y.item(i) - y_reg.item(0)) ** 2
	return numpy.sqrt(error * 1.0 / len(validation_x))

# Generate random lambdas and corresponding thetas
lambdas = [random.uniform(0.0, 150.0) for x in range(150)]
thetas = [ linreg(train_x, train_y, reg=l) for l in lambdas ]

def find_optimal_reg():	
	mses = [mse(wt) for wt in thetas]	
	min_index = mses.index(min(mses))
	return lambdas[min_index]

### Part C ###

def part_c():
	# Plot 1
	thetas_mag = [ numpy.linalg.norm(t) for t in thetas ]
	plt.plot(lambdas, thetas_mag, '8')

	# plot 2
	mses = [mse(wt) for wt in thetas]
	plt.plot(lambdas, mses, '^')

	plt.show()

# part_c()

### Part D ###

train_x_d = train_x[:,1:]
val_x_d = validation_x[:,1:]

def part_d(X, y):
	eye = numpy.eye(X.shape[0])
	one_matrix = numpy.ones(X.shape[0])
	reg_optimal = find_optimal_reg()

	X = numpy.matrix(X)
	y = numpy.matrix(y)

	A_mod = X.transpose() * (eye - one_matrix / X.shape[0])
	theta_opt = numpy.linalg.solve(A_mod * X + reg_optimal * numpy.eye(A_mod.shape[0]), A_mod * y)
	b_opt = sum((train_y - train_x_d * theta_opt)) / X.shape[0]
	original = linreg(train_x, train_y, reg_optimal)

	b_diff = abs((original[0] - b_opt).item(0))
	theta_diff = numpy.linalg.norm(theta_opt - original[1:])
	return b_diff, theta_diff

# print part_d(train_x_d, train_y)

### Part E ###

def part_e(X_train, X_val):
	X_train = numpy.matrix(X_train)
	Y_train = numpy.matrix(train_y)
	X_val = numpy.matrix(X_val)

	shape = (X_train.shape[1], 1)
	n = X_train.shape[0]
	err = 1e-6
	max_train_iters = 150
	linreg_theta = 2.5e-12
	linreg_b = 0.2
	reg_optimal = find_optimal_reg()

	eye = numpy.eye(X_train.shape[0])
	one_matrix = numpy.ones(X_train.shape[0])
	theta_opt = linreg(train_x, train_y, reg=reg_optimal)

	theta_ = numpy.zeros(shape)
	b_ = 0.

	grad_theta = numpy.ones_like(theta_)
	grad_b = numpy.ones_like(b_)

	objective_train = []
	objective_val = []

	print('Train:')
	while numpy.linalg.norm(grad_theta) > err and numpy.abs(grad_b) > err and len(objective_train) < max_train_iters:
		objective_train.append(
			numpy.sqrt(
				numpy.linalg.norm((X_train * theta_).reshape(-1,1) + b_ - train_y,) ** 2 / train_y.shape[0]
			)
		)
		
		objective_val.append(
			numpy.sqrt(
				numpy.linalg.norm((X_val * theta_).reshape(-1,1) + b_ - validation_y,) ** 2 / validation_y.shape[0]
			)
		)

		grad_theta = (
			(X_train.transpose() * X_train + reg_optimal * numpy.eye(shape[0])) * theta_ + X_train.transpose() * (b_ - train_y)
		) / X_train.shape[0]
		
		grad_b = (
			(X_train * theta_).sum() - train_y.sum() + b_ * n
		) / X_train.shape[0]

		theta_ = theta_ - linreg_theta * grad_theta
		b_ = b_ - linreg_b * grad_b

		if len(objective_train) % 25 == 0:
			print('-- finishing iteration {} - objective {:5.4f} - grad {}'.format(
				len(objective_train), objective_train[-1], numpy.linalg.norm(grad_theta)
			))
	
	print('==> Distance between intercept and orig: {}'.format(numpy.abs(theta_opt.item(0) - b_).item(0)))
	print('==> Distance between theta and original: {}'.format(
		numpy.linalg.norm(theta_ - theta_opt[1:])
	))

# train_x_e = train_x[:,1:]
# val_x_e = validation_x[:,1:]
# part_e(train_x_e, val_x_e)