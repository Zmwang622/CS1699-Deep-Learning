import collections
import random
import numpy as np
import matplotlib.pyplot as plt

"""
First, write a function forward that takes inputs X, W1, W2 and outputs y_pred, Z. 

This function computes activations from the front towards the back of the network, using 
fixed input features and weights.

Also use the forward pass function to evaluate your network after training.

Args:
X - a NxD matrix, N is # of samples, D is # of feature dimensions
W1 - MxD matrix representing weights between first and second layers
W2 - 1xM matrix representing weights between second and third layer

Returns: 
y_pred - Nx1 vector containing outputs at last layer of all N Samples
Z - NxM matrix containing the activations for all M hidden neurons for all N samples 
"""
def forward(X, W1, W2):
	z = np.tanh(X @ W1.T)
	# print(z.shape)
	# print(W2.shape)
	y_pred = z @ W2.T

	return (z,y_pred)

def generate_random_numbers(num_rows=1000000, num_cols=1, mean=0.0, std=.3):
  """Generates random numbers using `numpy` library.

  Generates a vector of shape 1000000 x 1 (one million by one) with random
  numbers from Gaussian (normal) distribution, with mean of 0 and standard
  deviation of 5.

  Note: You can use `num_rows`, `num_cols`, `mean` and `std` directly so no need
  to hard-code these values.

  Hint: This can be done in one line of code.

  Args:
    num_rows: Optional, number of rows in the matrix. Default to be 1000000.
    num_cols: Optional, number of columns in the matrix. Default to be 1.
    mean: Optional, mean of the Gaussian distribution. Default to be 0.
    std: Optional, standard deviation of the Gaussian dist. Default to be 5.

  Returns:
    ret: A np.ndarray object containing the desired random numbers.
  """
  # ret = None

  # Delete the following line and complete your implementation below.
  # raise NotImplementedError
  # All your changes should be above this line.
  return np.random.normal(mean,std,(num_rows,num_cols))

"""
Second, write a function backward that takes inputs X, y, M, iters, eta and outputs W1, W2, error_over_time. 

This function performs training using backpropagation (and calls the activation computation function as it iterates). 

Construct the network in this function, i.e. create the weight matrices and initialize the weights to small 
random numbers, then iterate: pick a training sample, compute the error at the output, then backpropagate to the hidden 
layer, and update the weights with the resulting error.

Args:
X - a NxD matrix of feeatures where N is # of samples and D is # of feature dimensions
y - a Nx1 vector y containing the ground-truth labels for the N samples
M - scalar describing the # of hidden neurons to use
iters - scalar describing how many interations to run (one sample used in each)
eta - scalar describing the learning rate

Returns:
W1 - MxD matrix representing new weights between first and second layers
W2 - 1xM matrix representing new weights between second and third layer
error_over_time - itersx1 vector that contains error on the sample used in each iteration
"""
def backward(X, y, M, iters, eta):
	W1 = generate_random_numbers(num_rows = M,num_cols = X.shape[1])
	W2 = generate_random_numbers(num_rows = 1,num_cols = M)
	index_set = set()
	error_over_time = []

	for _ in range(iters):
		(z, y_pred) = forward(X,W1,W2)
		index = random.randint(0,X.shape[0] - 1)
		delta_k = y_pred[index] - y[index]
		error_over_time.append(np.sqrt(((y_pred - y)**2).mean()))
		delta_j = []
		# W2 = W2.reshape((30,1))
		# print(W2[0][1])
		zn = z[index]
		for m in range(M):
			# print("Error derivative", (1-zn[m]**2))
			# print("Delta k: ", delta_k)
			# print("Weight", W2[m])
			# print("Product: ", delta_k * (1-zn[m]**2)*W2[m])
			delta_j.append((1-zn[m]**2) * delta_k * W2[0][m])
		
		delta_j = np.array(delta_j)
		for m,weight in np.ndenumerate(W2):
			weight = weight - eta * z[index][m[0]] * delta_k
			W2[0][m[0]] = weight

		# W1 is M x D
		# for m,row in np.ndenumerate(W1):
		# 	print(m)
			# for d,weight in np.ndenumerate(row):
			# 	weight = weight - eta * X[index][d] * delta_j[m[0]]
			# 	W1[m][d] = weight
		for m in range(len(W1)):
			row = W1[m]
			for d in range(len(row)):
				weight = W1[m][d]
				# print(delta_j[m])
				# print(X[index][d])
				weight = weight - eta * X[index][d] * delta_j[m]
				# print(weight)
				W1[m][d] = weight
	return (W1,W2,error_over_time)

# consts
PATH_TO_FILE = 'winequality-red.csv'
NUM_HIDDEN_UNITS = 30
NUM_ITERATIONS = 1000
LEARNING_RATE = 0.05

#receive data
my_data = np.genfromtxt(PATH_TO_FILE,delimiter = ';')[1:]

# test,train split
np.random.shuffle(my_data)
test,train = my_data[:len(my_data)//2] , my_data[len(my_data)//2:]
train_features,train_label = np.delete(train,-1,axis=1),train[:,-1]
test_features, test_label = np.delete(test,-1,axis=1),test[:,-1]

# print(train_features)
# print("Train features shape: ",train_features.shape)
# print(train_label)
# print("Train label shape: ",train_label.shape)

# standardization
mean = np.mean(train_features, axis = 0)
std = np.std(train_features, axis = 0)
train_features = (train_features - mean) / std
test_features = (test_features - mean) / std

# print("test features shape: ",test_features.shape)

# append a column of one
n = train_features.shape[0]
train_col = np.ones(n).reshape((n,1))
q = test_features.shape[0]
test_col = np.ones(q).reshape((q,1))
train_features = np.append(train_features,train_col,axis = 1)
test_features = np.append(test_features,test_col,axis = 1)

# print(train_features)
# print("Train features shape: ",train_features.shape)
# print(test_features)
# print("test feautres shape: ",test_features.shape)

(W1,W2,error_over_time) = backward(train_features,train_label,M = NUM_HIDDEN_UNITS, iters = NUM_ITERATIONS, eta = LEARNING_RATE)

(z,y_pred) = forward(test_features,W1,W2)

# print(y_pred)
# print(test_label)
mean_squared_error = np.sqrt(((y_pred - test_label)**2).mean())	

print(mean_squared_error)
plt.plot(error_over_time)
plt.show()
plt.close()