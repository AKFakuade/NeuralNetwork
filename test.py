import numpy as np
import nnfs
import math
from nnfs.datasets import spiral_data
nnfs.init()
e = math.e

# in the class, weights and biases are initialized for a neural network. range of -1 to 1
class Layer_Dense:
    # this is a constructor. self is like "this" in java.   
    def __init__(self, n_inputs, n_neurons):
         # initialize weights of a layer to be random numbers with size inputs (columns aka values per neuron) x neurons (rows)
         # multiply by 0.10 to ensure values are less than 1
         # notice that argument is columns x rows instead of rows x columns like it is in linear. 
         # that's so you don't have to transpose when calculating every dot product.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # each neuron has its own singular bias. this makes a 1 x neurons matrix.
        # biases are initialized to all zeroes at first. 
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# activation functions
# steps, sigmoids, and reLUs are used to approximate any 
# type of function that linear functions can't.
class Activation_ReLU:
     def forward(self, inputs):
          self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class Loss: 
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Categorical(Loss):
    def forward (self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
X, y = spiral_data(samples = 100, classes = 3)


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
loss_function = Loss_Categorical()
loss = loss_function.calculate(activation2.output, y)
print("loss: ", loss)



# make your first layer. 2 elements per neuron, 5 neurons
# number of neurons can be any. elements per neuron must match with the original input. so that dot product can work.
# layer1 = Layer_Dense(2, 5)
# # activation1 will turn all negative values to zeroes
# activation1 = Activation_ReLU()
#initialize layer 2. number of neurons in layer1 must match elements per neuron in layer2.

# pass in X as an input and forward it.
# forwarding it computes the above dot product and adds the vector of biases. (in output!)
# layer1.output will be the output of the dot products of all 5 neurons.
# layer1.forward(X)
# activation1.forward(layer1.output)


# # each neuron will have its own set of weights
# weights =   [[0.2, 0.8, -0.5, 1.0], 
#             [0.5, -0.91, 0.26, -0.5], 
#             [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# weights2 =   [[0.1, -0.14, 0.5], 
#             [-0.5, 0.12, -0.33], 
#             [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# # dot product. first argument determines type of output. (vector vs matrix)

# # dot product: 3x4 and 3x4. This doesn't even work in regular linear. dot (axb, bxc). (cols1 must equal rows2). result is axa
# # transpose weights. 
# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# #print(layer1_output)

# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# #print(layer2_output)





# # output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1, 
# #            inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2, 
# #            inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

# # print(output)




