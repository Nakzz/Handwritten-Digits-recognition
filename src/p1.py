import numpy as np
import pandas as pd
from numpy.linalg import norm

# logistic regresion nueral network class
class logisticRegressionNueralNet:
    
    def __init__(self, trainingData, numberOfLayes, outputLayerSize):

        self.alpha = 0.1; # learning rate
        self.epsilon = 1;# epsilon
        self.maxIter =1000
        self.costs = [self.maxIter]

        self.numberOfInstances, numberOfFeatures = trainingData.shape;
        self.numberOfFeatures = numberOfFeatures -1 # features are total x dim -1

        self.xFeatures = (trainingData.loc[:, 1:])/255 # index [1:] of every row

        
        # print(self.xFeatures)
        # print(self.xFeatures/255)

        self.y = (trainingData.loc[:, 0]).values # index 0 of every row
        self.biases = np.random.uniform(low=-1.0, high=1.0, size=(1, self.numberOfInstances)) # random bias between 1 and -1, for all instances
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(1, self.numberOfFeatures)) # random weights between 1 and -1 based on all features
        self.outputLayer = np.zeros((outputLayerSize, 1))

        # print("weights size: " ,self.weights.shape)
        # print("xFeatures size: " , self.xFeatures.shape)
        # print("y size: " , self.y.shape)
        # print("biases size: " , self.biases.shape)

        # for i in range(numberOfInstances+1):
        #     print(i)

    def getResults(self):
        return [self.weights, self.biases, self.outputLayer]

    #updates weights and biases
    def trainModel(self):
        #evaluate ai initial
        a_i = self.sigmoidActivation(self.prediction_A(self.weights, self.xFeatures, self.biases))
        
        print()
        print("train model:")
        print("\t a_i size: " , a_i.shape)
        print("\t y size: " , self.y.shape)

        actualCosts = self.cost(self.y, a_i)
        actualCost = norm(actualCosts, 1)
        currentCost = 0

        
        # print("\t currentCosts size: " ,currentCosts.shape)
        print("\t currentCost: " , actualCost)
        
        iteration = 0;

        while(abs(actualCost - currentCost) > self.epsilon):
            
            if(iteration > self.maxIter ):
                print("Max iteration reached")
                break

            print("\t Training iteration: ", iteration +1)
            
            #store cost
            self.costs.append(currentCost)

            #compute gradient of cost_w
            grad_cost_w = self.gradient_cost_w(self.y, a_i, self.xFeatures)

            oldWeights = self.weights
            # print(oldWeights)
            #update the weights and biases 
            self.weights = self.weights - self.vect_scalar_multiplication(grad_cost_w, self.alpha) #self.alpha*(grad_cost_w)
            self.biases = self.biases - self.vect_scalar_multiplication(self.gradient_cost_b(self.y, a_i), self.alpha) #self.alpha*(self.gradient_cost_b)


            # print((self.weights - oldWeights))

            #update currentCost
            #evaluate ai 
            a_i = self.sigmoidActivation(self.prediction_A(self.weights, self.xFeatures, self.biases))
            currentCosts = self.cost(self.y, a_i)
            currentCost = norm(currentCosts, 1)

            # print("\t \t updated weights: " ,self.weights)
            print("\t \t actualCost: " , actualCost)
            print("\t \t updated currentCost: " , currentCost)

            iteration = iteration +1;

        print()

    #tests model with a set of x_features
    def testModel(self, features):
        # do some fancy math to update the outputLayer, PS: turns out not so fancy
        self.outputLayer = self.prediction_A(self.weights, features, self.biases)

        return self.outputLayer

# define helper methods
    #vectorize scaler multiplicatin
    def vect_scalar_multiplication(self, vector, scalar):
        function = lambda x : x * scalar
        vSig = np.vectorize(function)
        computed = vSig(vector)
        return computed 

    # prediction ai
    def prediction_A(self, w, x, b):

        # print()
        # print("prediction_A input:")
        # print("\t weights size: " ,w.shape)
        # print("\t xFeatures size: " , x.shape)
        # print("\t biases size: " , b.shape)
        # print()

        return (np.dot(w,x.T)+b)

    # sigmoid activation function
    def sigmoidActivation(self, z):

        print("sigmoid input:", z.shape)
        print(z)


        sigmoid = lambda x : np.around((1.0/(1.0+ np.exp(-x))), decimals=5)
        vSig = np.vectorize(sigmoid)
        activated = vSig(z)

        print("activated: ",activated)


        return activated
    
    def sigmoid_derivative(self, x):
        return self.sigmoidActivation(x) * (1.0- self.sigmoidActivation(x))

    # cost function sqaure distance
    def cost(self, y, a):
        return ((y - a) - (self.vect_scalar_multiplication( y * a, 2)) + (y - a))

    # # gradient of cost respect to w
    def gradient_cost_w(self, y, a, x):

        # print()
        # print("gradient_cost_w:")
        # print("\t a size: " , a.shape)
        # print("\t y size: " , y.shape)
        # print("\t x size: " , x.shape)
        
        x = x.values

        # print(a, type(a))
        # print(y, type(y))
        # # print(y - a)
        # print(x.T, type(x.T))

        # print("\t y-a:", (y-a).shape)

        grad = np.matmul((y-a), x )

        # print("grad: ", grad.shape)
        # print(grad)

        return self.vect_scalar_multiplication(grad, .5)

     # gradient of cost respect to b
    def gradient_cost_b(self, y, a,):
        return (a - y)

    def partial_cost_a(a, y):
        return (a - y);
    
    def gradient_ai_w(self):
        return self.sigmoid_derivative()




# train using mnist_train.csv
# read csv and insert in array
# mnist file format label, pix-11, pix-12, pix-13, ...
# mnistTest = pd.read_csv('../data/mnist_train.csv', header=None)
mnistTest = pd.read_csv('../data/mnist_train_small.csv', header=None)
numberOfInstances, numberOfFeatures = mnistTest.shape;

# xFeatures = mnistTest.loc[:, 1:] 

# print(numberOfInstances, numberOfFeatures)
# print(mnistTest.head(5))
# print(xFeatures.head(5))

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=2000)

neuralNet = logisticRegressionNueralNet(mnistTest, 3, 10);

neuralNet.trainModel()

result = neuralNet.getResults();


# for x in result:
#     print(x);

