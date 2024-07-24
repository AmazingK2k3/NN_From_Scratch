import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()



X,y = spiral_data(samples = 100, classes = 3)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        #set regularization strength
        self.weight_regularizer_l1 =  weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
   
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

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

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0.,momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations +=1


class Optimizer_Adagrad:
    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights/ (np.sqrt(layer.weight_cache)+ self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases/ (np.sqrt(layer.bias_cache)+ self.epsilon)    

    def post_update_params(self):
        self.iterations +=1

class Optimizer_RMSprop:
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights **2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases **2

        layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases/ (np.sqrt(layer.bias_cache)+ self.epsilon)    

    def post_update_params(self):
        self.iterations +=1

class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)  
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_mometums = self.beta_1 * layer.weight_momentums + (1-self.beta_1)* layer.dweights

        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations+1))


        layer.weight_cache = self.beta_2 * layer.weight_cache +  (1-self.beta_2) * layer.dweights **2
        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1-self.beta_2) * layer.dbiases **2

        weights_cache_corrected = layer.weight_cache / (1-self.beta_2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1-self.beta_2 ** (self.iterations+1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected/ (np.sqrt(weights_cache_corrected)+ self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected/ (np.sqrt(bias_cache_corrected)+ self.epsilon)    

    def post_update_params(self):
        self.iterations +=1



class Loss:
    def regularization_loss(self,layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 >0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 >0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 >0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 >0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    
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

optimizer = Optimizer_Adam(learning_rate = 0.05, decay=5e-7)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    #Perform a forward pass of the combined activation function and loss function
    #takes the output of second dense layer here and the true labels
    data_loss = loss_activation.forward(dense2.output,y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1)  + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    #loss_activation.output is the output from the softmax activation function  
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
         y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')


    #Backward pass
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    #Validated the model
    X_test, y_test = spiral_data(samples = 100, classes = 3)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y_test)

    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis = 1)
    accuracy = np.mean(predictions == y_test)

    print(f'Validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
    



