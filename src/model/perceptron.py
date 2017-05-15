# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.1, epochs=30):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.
        
        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            for idx in range(len(self.trainingSet.input)):
                out = int(self.fire(self.trainingSet.input[idx]))
                err = self.trainingSet.label[idx] - out
                self.updateWeights(self.trainingSet.input[idx], err)

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        """
        Perceptron Learning Algorithm 
        w_{ij}(k+1)=w_{ij}(k) bei korrekter Ausgabe,
        w_{ij}(k+1)=w_{ij}(k)+\alpha x_{i} bei Ausgabe 0 und gewünschter Ausgabe 1 
        w_{ij}(k+1)=w_{ij}(k)-\alpha x_{i} bei Ausgabe 1 und gewünschter Ausgabe 0

        """

        self.weight + error * self.learningRate*np.array(input)

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
