import numpy as np
import pandas as pd
from numpy.linalg import norm
# logistic regresion nueral network class
class logisticRegressionNueralNet:
    
    def __init__(self, trainingData, numberOfLayes, outputLayerSize):

        self.alpha = 0.1; # learning rate
        self.epsilon = 1;# epsilon

        self.numberOfInstances, numberOfFeatures = trainingData.shape;
        self.numberOfFeatures = numberOfFeatures -1 # features are total x dim -1

        self.xFeatures = trainingData.loc[:, 1:] # index [1:] of every row
        self.y = trainingData.loc[:, 0] # index 0 of every row
        self.biases = np.random.uniform(low=-1.0, high=1.0, size=(self.numberOfInstances,1)) # random bias between 1 and -1, for all instances
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(1, self.numberOfFeatures)) # random weights between 1 and -1 based on all features
        self.outputLayer = np.zeros((outputLayerSize, 1))

        print("weights size: " ,self.weights.shape)
        print("xFeatures size: " , self.xFeatures.shape)
        print("y size: " , self.y.shape)
        print("biases size: " , self.biases.shape)
        # for i in range(numberOfInstances+1):
        #     print(i)

    def getResults(self):
        return [self.weights, self.biases, self.outputLayer]

    #updates weights and biases
    def trainModel(self):
        #evaluate ai
        a_i = self.sigmoidActivation(self.prediction_A(self.weights, self.xFeatures, self.biases))
        
        print(a_i)

        # while(currentCost - prevCost < episilon)

            #compute gradient of cost_w

            #update the weights and biases w = w - self.alpha*(gradient_cost_w)
            # b = b = self.alpha*(gradient_cost_b)

    #tests model with a set of x_features
    def testModel(self, features):
        # do some fancy math to update the outputLayer, PS: turns out not so fancy
        self.outputLayer = self.prediction_A(self.weights, features, self.biases)

        return self.outputLayer

# define helper methods

    # prediction ai
    def prediction_A(self, w, x, b):
        return (np.dot(w,x.T)+b)

    # sigmoid activation function
    def sigmoidActivation(self, x):
        return 1.0/(1.0+ np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoidActivation(x) * (1.0- self.sigmoidActivation(x))

    # cost function sqaure distance
    def cost(self, y, a):
        return ((y - a) * (y - a))

    # gradient of cost respect to w
    # def gradient_cost_w(self, y, a,):
        # return (1/(2*self.numberOfInstances)) * ((y - a) * )

# delta_y computation: delta_w, delta_b

# gradient descent 

# matrix multiplication





# define two numpy arrays
# a = np.array([[1,2],[3,4]])
# b = np.array([[1,1],[1,1]])
# print(a)

# print(b)


# train using mnist_train.csv
# read csv and insert in array
# mnist file format label, pix-11, pix-12, pix-13, ...
mnistTest = pd.read_csv('../data/mnist_test.csv', header=None) 
numberOfInstances, numberOfFeatures = mnistTest.shape;

# xFeatures = mnistTest.loc[:, 1:] 

# print(numberOfInstances, numberOfFeatures)
# print(mnistTest.head(5))
# print(xFeatures.head(5))

neuralNet = logisticRegressionNueralNet(mnistTest, 3, 10);

neuralNet.trainModel()

result = neuralNet.getResults();


# for x in result:
#     print(x);

