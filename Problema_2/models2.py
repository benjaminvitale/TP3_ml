import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import patches


import numpy as np

import numpy as np

class LDA:
    def __init__(self):
        self.means_ = None  # Media de cada clase
        self.priors_ = None  # Probabilidad a priori de cada clase
        self.covariance_ = None  # Matriz de covarianza compartida
        self.classes_ = None  # Clases únicas en y

    def fit(self, X, y):
        """
        Ajusta el modelo LDA a los datos.
        X: Matriz de diseño (n_samples, n_features)
        y: Vector de etiquetas (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)  # Encuentra las clases únicas en y
        
        n_classes = len(self.classes_)

        # Inicializar medias, covarianza y probabilidades a priori
        self.means_ = np.zeros((n_classes, n_features))
        self.covariance_ = np.zeros((n_features, n_features))
        self.priors_ = np.zeros(n_classes)

        # Calcular medias y probabilidades a priori
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.priors_[idx] = X_c.shape[0] / n_samples
        
        # Calcular la matriz de covarianza compartida
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.covariance_ += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        self.covariance_ /= (n_samples - n_classes)  # Covarianza ponderada

    def _gaussian_density(self, X, mean, covariance):
        """Calcula la densidad gaussiana para la distribución multivariada."""
        n_features = X.shape[1]
        cov_inv = np.linalg.inv(covariance)
        norm_factor = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(covariance))
        diff = X - mean
        exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        return np.exp(exponent) / norm_factor

    def predict_proba(self, X):
        """
        Devuelve las probabilidades de pertenencia a cada clase (n_samples, n_classes)
        X: Matriz de diseño (n_samples, n_features)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        posteriors = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes_):
            prior = np.log(self.priors_[idx])
            likelihood = self._gaussian_density(X, self.means_[idx], self.covariance_)
            posteriors[:, idx] = prior + np.log(likelihood)

        # Convertir las posteriors en probabilidades usando softmax
        posteriors = np.exp(posteriors)
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)

        return posteriors

    def predict(self, X):
        """
        Predice la clase más probable para cada muestra (n_samples,)
        Devuelve un array de tamaño (n_samples,) con los valores 0, 1 o 2.
        """
        posteriors = self.predict_proba(X)
        predicted_classes = np.argmax(posteriors, axis=1)  # Devuelve la clase con la mayor probabilidad
        for i in range(len(posteriors)):
            if posteriors[i][1] > 0.06:
                predicted_classes[i] = 1
        return predicted_classes



class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self,max_depth,min_samples_leaf, min_information_gain,min_part_entropy) -> None:
        """
        Constructor function for DecisionTree instance
        Inputs:
            max_depth (int): max depth of the tree
            min_samples_leaf (int): min number of samples required to be in a leaf 
                                    to make the splitting possible
            min_information_gain (float): min information gain required to make the 
                                          splitting possible                              
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.min_part_entropy = min_part_entropy

    def entropy(self, class_probabilities: list) -> float: #esto podria ser indice de gini o chi cuadrado tambien
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        """
        Calculates the entropy of a partitioned dataset. 
        Inputs:
            - subsets (list): list of label lists 
            (Example: [[1,0,0], [1,1,1] represents two subsets 
            with labels [1,0,0] and [1,1,1] respectively.)

        Returns:
            - Entropy of the labels
        """
        # Total count of all labels across all subsets.
        total_count = sum([len(subset) for subset in subsets]) 
        # Calculates entropy of each subset and weights it by its proportion in the total dataset 
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Partitions the dataset into two groups based on a specified feature 
        and its corresponding threshold value.
        Inputs:
        - data (np.array): training dataset
        - feature_idx (int): feature used to split
        - feature_val (float): threshold value 
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
        
    def find_best_split(self, data: np.array) -> tuple:
        """
        Finds the optimal feature and value to split the dataset on 
        at each node of the tree (with the lowest entropy).
        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - 2 splitted groups (g1_min, g2_min) and split information 
            (min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy)
        """
        min_part_entropy = 1e9
        feature_idx =  list(range(data.shape[1]-1))

        for idx in feature_idx: # For each feature
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) # Calc 25th, 50th, and 75th percentiles
            for feature_val in feature_vals: # For each percentile value we partition in 2 groups
                g1, g2, = self.split(data, idx, feature_val)
                part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]]) # Calculate entropy of that partition
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def find_label_probs(self, data: np.array) -> np.array:
        """
        Computes the distribution of labels in the dataset.
        It returns the array label_probabilities, which contains 
        the probabilities of each label occurring in the dataset.

        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - label_probabilities (np.array): numpy array with the
            probabilities of each label in the dataset.
        """
        # Transform labels to ints (assume label in last column of data)
        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> Node:
        """
        Recursive, depth first tree creation algorithm.
        Inputs:
            - data (np.array): numpy array with training data
            - current_depth (int): current depth of the recursive tree
        Returns:
            - node (Node): current node, which contains references to its left and right child nodes.
        """
        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)
        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)
        # Calculate information gain
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        # Create node
        node = Node(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
        
        return node
    
    def predict_one_sample(self, X: np.array) -> np.array:
        """
        Returns prediction for 1 dim array.
        """
        node = self.tree
        # Finds the leaf which X belongs to
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets.
        Inputs:
            - X_train (np.array): training features
            - Y_train (np.array): training labels
        """
        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        # Create tree
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Returns the predicted probs for a given data set
        """
        if len(X_set.shape) == 1:
        # Si X_set es un array unidimensional, asegúrate de que tenga la forma correcta
            X_set = X_set.reshape(1, -1)
        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Returns the predicted labels for a given data set
        """
        pred_probs = self.predict_proba(X_set)

        preds = np.argmax(pred_probs, axis=1)
        
        return preds   




class RandomForest:
    def __init__(self, n_estimators, max_depth,min_sample_leaf,min_information_gain,entropy):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_information_gain = min_information_gain
        self.entropy = entropy
        
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)  # Cambiar a n_samples
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Crear un árbol de decisión y ajustar
            tree = DecisionTree(self.max_depth, self.min_sample_leaf, self.min_information_gain, self.entropy)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X):
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        #print(tree_preds.shape)
        return np.array([np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])])

    def predict_proba(self, X):
        """
        Calcula las probabilidades de pertenecer a cada clase para cada ejemplo en X.
        X: matriz de datos (n_samples, n_features)
        Return: matriz (n_samples, n_classes) de probabilidades para cada clase
        """
        # Obtener las predicciones de todos los árboles
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_estimators, n_samples)
        
        # Número de clases distintas
        n_classes = 3
        # Inicializar la matriz de probabilidades
        proba_matrix = np.zeros((X.shape[0], n_classes))
        
        # Contar las predicciones para cada clase
        for i in range(X.shape[0]):
            class_counts = np.bincount(tree_preds[:, i], minlength=n_classes)  # Contar ocurrencias de cada clase
            proba_matrix[i, :] = class_counts / self.n_estimators  # Normalizar para obtener las probabilidades

        return proba_matrix

class LogisticRegressionMulticlass:
    def __init__(self, max_iter, learning_rate, l2):
        """
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        l2: L2 regularization term
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l2 = l2
        self.classifiers = []  # Lista de clasificadores binarios, uno por clase

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
    
    def fit(self, X, y, class_weights=None):
        """
        Fits the logistic regression model to the data points using One-vs-Rest strategy.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        class_weights: dictionary of class weights, for rebalancing
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)  # Identificar las clases únicas en los datos
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Crear un clasificador binario por cada clase
        self.classifiers = []

        for class_idx in self.classes_:
            # Inicializar coeficientes para esta clase
            coef_ = np.zeros(n_features)
            for _ in range(self.max_iter):
                # Crear etiquetas binarias: 1 si es la clase actual, 0 de lo contrario
                y_binary = (y == class_idx).astype(int)

                # Predicción de probabilidad
                z = np.dot(X, coef_)
                y_hat = self._sigmoid(z)
                
                # Factor de re-balanceo si se especifica
                if class_weights is not None and class_idx in class_weights:
                    pi_1 = np.mean(y_binary == 1)
                    pi_2 = np.mean(y_binary == 0)
                    C = pi_2 / pi_1 if pi_1 > 0 else 1
                    weights = np.where(y_binary == 1, C, 1)
                else:
                    weights = np.ones(y_binary.shape)

                # Gradiente de la función de pérdida con regularización L2
                gradient = np.dot(X.T, weights * (y_hat - y_binary)) / y.size
                gradient[1:] += (self.l2 / y.size) * coef_[1:]  # L2 regularization

                # Actualizar los coeficientes
                coef_ -= self.learning_rate * gradient

            # Guardar los coeficientes del modelo para esta clase
            self.classifiers.append(coef_)

    def predict_proba(self, X):
        """
        Predicts probabilities for each class for inputs X using the trained classifiers.
        X: design matrix (n_samples, n_features)
        """
        X = self._add_intercept(X)
        
        # Obtener las probabilidades para cada clase
        probas = np.array([self._sigmoid(np.dot(X, coef)) for coef in self.classifiers]).T
        
        return probas

    def predict(self, X):
        """
        Predicts the class for the inputs X using the trained classifiers.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        
        # Elegir la clase con la mayor probabilidad predicha
        n= []
        for i in probas:
            if i[1] >= 0.09:
                n.append(1)
            if i[2] >= 0.4 and i[1] < 0.09:
                n.append(2)
            if i[1] < 0.09 and i[2] < 0.4:
                n.append(0)
        #return np.argmax(probas, axis=1)
        return n
