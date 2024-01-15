import numpy
import random
import math
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
### GENERATING DATA
classA = np.concatenate((np.random.randn(10,2) * 0.2 + [1.5, 0.5], np.random.randn(10,2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20,2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


# kernel function linear
def linear_kernel(x, y):
    return np.dot(x, y)

# kernel function polynomial
def polynomial_kernel(x, y, p=3):
    return (np.dot(x, y) + 1) ** p

# kernel function rbf
def rbf_kernel(x, y, sigma=2.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# implement zerofun
def zerofun(alpha):
    return np.dot(alpha, targets)

Kernel = linear_kernel

# implement objective function

# implement P matrix
Pmatrix = numpy.zeros((N,N))
for i in range(N):
    for j in range(N):
        Pmatrix[i][j] = targets[i] * targets[j] * Kernel(inputs[i], inputs[j])

def objective(alpha):
    #Define a function which implements equation (4).
    # This function will only receive the vector α⃗ as a parameter.
    # You can use global variables for other things that the function needs (t and K values).
    return 0.5 * np.dot(alpha.T, numpy.dot(Pmatrix, alpha)) - np.sum(alpha)

# N is the number of training samples
start = np.zeros(N)
# B = [(0,C) for b in range(N)]
B = [(0,None) for b in range(N)] # if no upper bound

XC = {'type': 'eq', 'fun':zerofun}

ret = minimize(objective, start, bounds=B, constraints=XC)
if (not ret['success']): # The string 'success' instead holds a boolean representing if the optimizer has found a solution
    raise ValueError('Cannot find optimizing solution')

# extract the non zero alpha values
alpha = ret['x']
non_zero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 1e-5]

# calculate the b value
b = 0
for n in range(len(non_zero)):
    b += non_zero[n][2]
    b -= np.sum([non_zero[m][0] * non_zero[m][2] * Kernel(non_zero[m][1], non_zero[n][1]) for m in range(len(non_zero))])
b /= len(non_zero)

#implement indicator function
def indicator(x, y):
    return np.sign(np.sum([n[0] * n[2] * Kernel(np.array([x, y]), n[1]) for n in non_zero]) - b)

### PLOTTING

# plot the samples
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal') # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save a copy in a file
plt.show()  # Show the plot on the screen

# plot the decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))


