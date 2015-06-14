# coding: utf-8

from neurolab.net import newff
from neurolab.train import train_gd


class NeuralNetwork:

    def __init__(self, nodes):
        self.nodes = nodes

    def reshape(self, data):
        size = len(data)
        return data.reshape(size, 1)

    def fit(self, X, y, epochs=300, learning_rate=0.2):
        """
        :Parameters:
            X: input data
            y: target data
        """

        minmax = [[X.min(axis=0)[i], X.max(axis=0)[i]] for i in xrange(0, len(X[0]))]
        size = [self.nodes, 1]
        self.ann = newff(minmax, size)

        tar = self.reshape(y)

        return train_gd(self.ann, X, tar, epochs=epochs, show=0, goal=learning_rate)

    def predict(self, X):
        return self.ann.sim(X)