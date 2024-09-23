import numpy as np
class LogisticRegression:
    def __init__(self, threshold, max_iter, learning_rate,l2):
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
    
    def fit(self, X, y,c):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        

        pi_1 = np.mean(y == 1)  # Probabilidad de la clase 1 (minoritaria)
        pi_2 = np.mean(y == 0)  # Probabilidad de la clase 0 (mayoritaria)
        
    # Factor de re-balanceo
        C = pi_2 / pi_1
        weights = np.where(y == 1, C, 1)
        # Initialize the coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predict probability
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # NLL gradient
            if c == 1:
                gradient = np.dot(X.T, weights * (y_hat - y)) / y.size
            else:
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
        #return prob_positive
    def predict(self, X):
        n = []
        """
        Predicts class (0 or 1) for the inputs X using a threshold.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        #prob_negative = 1 - probas
        #n = np.vstack((prob_negative, probas)).T
        """
        for i in probas:
            if i[0] >= self.threshold:
                n.append(1)
            else: 
                n.append(0)
        return n"""
        return (probas >= self.threshold).astype(int)