import numpy as np

class Logistic_Regression():

    # Declaring Learning rate and number of iterations (Hyperparameters)
    def __init__(self,learning_rate,no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
    

    # Fit function to train the model with dataset
    def fit(self,X,Y):
        
        # The Matrix shape of mXn of the data set m is the number of rows and n is the number of columns 
        self.m ,self.n = X.shape
        
        #Initializing weight and bias 
        self.w = np.zeros(self.n)
        
        self.b = 0
        
        self.X = X
        
        self.Y = Y
        
        #Implementing Gradient Descent for optimization 
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        
        # Y_hat formula (Sigmoid Function)
        Y_hat = 1 / ( 1 + np.exp(- ( self.X.dot(self.w) + self.b) ))  #wX+b
        
        
        # Derivatives 
        
        dw = (1/self.m) * np.dot(self.X.T , (Y_hat - self.Y))   # Note we have taken Transformations of X as X.T
        
        db = (1/self.m) * np.sum(Y_hat - self.Y)
        
        # Updating the weights and bias 
        self.w = self.w  - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
        #Sigmoid Function and Decision Boundart
        
        
    def predict(self,X):
        
        Y_pred = 1 / ( 1 + np.exp(- ( X.dot(self.w) + self.b) ))
        Y_pred = np.where(Y_pred > 0.5 , 1 ,0)
        return Y_pred


