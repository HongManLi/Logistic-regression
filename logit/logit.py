import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class logis:
    '''A logistic regression class using vanilla gradient descent. Can only differentiate between two classes.'''
    
    def __init__(self, x, y, alpha=0.01, iterations=10000):
        ''' params numpy matrix x: input matrix where each row is a data sample while each column is a single attribute
            params 1d numpy matrix y: 1d matrix of labels we are interested in predicting
            params float alpha: learning rate
            params int iterations: number of iterations
        '''
        
        #Number of rows
        rows = x.shape[0]
        
        #Shape y
        y = y.reshape(len(y),1)
        
        #We initialize thetas as all ones
        self.thetas = np.ones((x.shape[1] + 1, 1)) 
        
        #We initialize x matrix by adding in extra column of ones for the intercept coefficient
        x = np.c_[np.ones(rows), x]
        
        #We need this for storiong our cost at each iteration
        self.cost = []
        
        for i in range(iterations):
            #Making our predictions - dot product between our x and our theta
            predictions = np.matmul(x, self.thetas)
            #Bounding our predictions with sigmoid transformation - this is probability of the instance
            predictions = 1/ (1 + np.exp(-1 * predictions))
            
            
            #The cost is given by a applying log to the predictions and comparing with actual y value - this is not used
            #in updating theta, it is only used for grphical depiction of changes in error as iterations increases
            self.cost.append(
                        sum( 
                            (np.multiply(-1 * y, np.log(predictions))) - (np.multiply((1 - y), np.log(1 - predictions))) 
                         )
                        )
            
            #Getting the difference between our predictions and actual y value
            difference = predictions - y
            #Updating our thetas
            self.thetas = self.thetas - (alpha/rows) * np.matmul(x.T, difference)

    
    def get_thetas(self):
        '''This will return the coefficients of our hypothesis (called thetas here)'''
        return self.thetas
    
    def plot_errors(self):
        '''This will return a plt plot of cost versus iterations'''
        return plt.plot([i + 1 for i in range(len(self.cost))], self.cost)
        
    def probabilities(self, x):
        '''
            Return predictions for input matrix x as an array of probabilities
           
            params numpy matrix x: this is the matrix of attributes which we want to generate labels for
        '''
        #Again we need extra column of ones for intercept coefficients
        x = np.c_[np.ones(x.shape[0]), x]
        probs = np.matmul(x, self.thetas)
        return 1/ (1 + np.exp(-1 * probs))
    
    def prediction(self, x, threshold=0.5):
        '''
            Return predictions for input matrix x
            
            params numpy matrix x: this is the matrix of attributes which we want to generate labels for
            params float threshold: cut off point for decidinng whether a sample point is a positive or negative sample
        '''
        #First we call our probabilities method to get the probability of a positive sample for each data point
        pred = self.probabilities(x)
        #Applying our threshold to the probabilities to get our predictions
        pred[pred > threshold] = 1
        pred[pred < threshold] = 0
        return pred
    
    def accuracy(self, x, y, threshold=0.5):
        '''
            Return the accuracy of our model as a decimal
            params numpy matrix x: this is the matrix of attributes which we want to generate labels for
            params 1d numpy matrix y: this is the actual labels of the data we are comparing our predictions against
            params float threshold: cut off point for decidinng whether a sample point is a positive or negative sample
        '''
        #First we call the prediction method to get our predictions
        predi = self.prediction(x, threshold)
        #Reshaping required to ensure y is a 1 dimensional array that can be compared with our predictions
        y = y.reshape(len(y), 1)
        return sum(predi == y)/y.shape[0]