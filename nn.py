import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()



X,y = spiral_data(samples = 100, classes = 3)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis =1 ,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
    
    def backward(self,dvalues):
        #Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1,1)
            #calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            #calculate sample-wise gradient
            #and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        #clipping the values to avoid log(0) which is undefined
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        #y_true is a 1D array of class indices hence we use range(samples) to get the indices of the correct confidences to pick the exact values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        #y_true is a one-hot encoded matrix of shape(samples,classes) hence we use element-wise multiplication and then sum along the axis 1

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        #number of of labls in every sample
        lables = len(dvalues[0])

        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(lables)[y_true]

        #calculate gradient    
        self.dinputs = -y_true / dvalues
        #normalize gradient
        self.dinputs = self.dinputs / samples

#Softmax classifier combined with Softmaxz activation
#and cross entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            #np.argmax returns the idnex of the max values, but needs to be done sample wise to get vector of indices.
            y_true = np.argmax(y_true, axis = 1)
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient - instead of performing the subtraction of the full arrays, we taking advantages of the fact that 
        #the y being Y_true is a one-hot encoded array, there is only a singular value of l in these vectors and remaining are zeroes.
        #Meaning we can index the prediction array with the sample number and its true value index, subtracting 1 from these values
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples

X,y = spiral_data(samples = 100, classes = 3)

# Create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()

#Create second dense layer with 64 input features  and 3 output values
dense2 = Layer_Dense(64,3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

#Perform a forward pass of the combined activation function and loss function
#takes the output of second dense layer here and the true labels
loss = loss_activation.forward(dense2.output,y)

print(loss_activation.output[:5])

print('Loss:',loss)
#loss_activation.output is the output from the softmax activation function  
predictions = np.argmax(loss_activation.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)

print('Accuracy:',accuracy)

#Backward pass
loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

#Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)


