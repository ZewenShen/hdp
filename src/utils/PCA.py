import numpy as np

class PCA:
    """Principal Component Analysis"""
    def __init__(self, X):
        """
        X: 2D sample matrix (paramter num * sample num). Each column is a sample.
        X doesn't have to be normalized.
        """
        assert len(X.shape) == 2, "PCA input X should be a 2D matrix"
        self.X = X
        self.dimension, sample_size = X.shape
        parameters_mean = [np.mean(x) for x in X]
        normalized_X = (X.T - parameters_mean).T
        covariance_mat = normalized_X.dot(normalized_X.T)/sample_size
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance_mat)
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues, self.eigenvectors = self.eigenvalues[idx], self.eigenvectors[:, idx].T

    def reduce_dimension(self, k):
        """
        k: The dimension of space that we are mapping X to.
        """
        assert k < self.dimension, "PCA target dimension should be smaller than the input data dimension"
        P = self.eigenvectors[:k, :]
        return P.dot(self.X)