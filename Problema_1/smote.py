import numpy as np

def mix(X, y):
    mixer = np.random.permutation(len(X))
    X_new = X[mixer]
    y_new = y[mixer]
    return X_new, y_new


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_nearest_neighbors(X, point, k=8):
    distances = []
    
    for i in range(len(X)):
        distance = euclidean_distance(point, X[i])
        distances.append((i, distance))
    
    distances.sort(key=lambda x: x[1])
    
    neighbors = [idx for idx, _ in distances[1:k+1]]  
    
    return neighbors


def smote(X, y, k=8):
    minority_class = X[y == 1]

    majority_class_size = np.sum(y == 0)
    minority_class_size = len(minority_class)
    num_new_samples = majority_class_size - minority_class_size

    new_samples = []
    for _ in range(num_new_samples):
        index = np.random.randint(0, len(minority_class))
        point = minority_class[index]

        neighbors = k_nearest_neighbors(minority_class, point, k)
        
        neighbor_idx = np.random.choice(neighbors)
        neighbor = minority_class[neighbor_idx]

        diff = neighbor - point
        new_sample = point + np.random.rand() * diff
        new_samples.append(new_sample)

    new_samples = np.array(new_samples)
    X_resampled = np.vstack((X, new_samples))
    y_resampled = np.hstack((y, np.ones(len(new_samples))))

    return mix(X_resampled, y_resampled)