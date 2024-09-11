import numpy as np
class LogisticRegression:
    def __init__(self, threshold=0.5, max_iter=1000, learning_rate=0.01,l2 = 0.1):
        """
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
        self.l2 = l2
    
    def _sigmoid(self, z):
        """
        Sigmoid function to transform inputs into probabilities.
        z: scalar or numpy array
        """
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """
        Adds column of 1s to X for the intercept (bias) term.
        X: input feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        # Initialize the coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predict probability
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # NLL gradient
            gradient = np.dot(X.T, (y_hat - y)) / y.size

            gradient[1:] += (self.l2 / y.size) * self.coef_[1:]
            # Update coefficients
            self.coef_ -= self.learning_rate * gradient
        
        self.intercept_ = self.coef_[0] # Intercept is the fist value of coef_
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class for inputs X.
        X: design matrix (n_samples, n_features)
        """
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T
    
    def predict(self, X):
        """
        Predicts class (0 or 1) for the inputs X using a threshold.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)