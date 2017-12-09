
"""
 Bag learner. @Author: Yuanzheng Zhu (yzhu319)
 Don't care what learner is bagged into the bag learner
 Regression --> Classification, 5 bags take averages--> 5 bags vote. Ypred_bag.mean[0] --> mode(Ypred_bag)[0][0]
"""

import numpy as np
from scipy.stats import mode

class BagLearner(object):

    def __init__(self, learner, kwargs, bags = 20, boost = False, verbose = False):
        self.bags = bags
        self.verbose = verbose
        self.learner = learner
        self.kwargs = kwargs
        self.learnersBag = []
        for i in range(0,bags):
            self.learnersBag.append(self.learner(**kwargs))

    def author(self):
        return 'yzhu319'

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # data used to build the tree; append dataY to dataX, N+1 column
        dataY = dataY.reshape((dataX.shape[0],1))
        data = np.concatenate((dataX, dataY), axis = 1) #concatenate horizontally

        if(self.verbose):print "original data is\n",data

        #!! sample with replacement, each data sample used for 1 learner in bag can be different
        #training set is of size N, each bag contain N items.some of the data items will be repeated.
        for i in range(0, self.bags):
            row_index = np.random.randint(data.shape[0], size= data.shape[0])
            data_shuffle = data[row_index,:]
            dataX_shuffle = data_shuffle[:,0:-1]
            dataY_shuffle = data_shuffle[:,-1]
            self.learnersBag[i].addEvidence(dataX_shuffle, dataY_shuffle)

        #for i in range(0, self.bags):
        #   self.learnersBag[i].addEvidence(dataX, dataY)

    def query(self,X):
        """
        @summary: query a bag learner.

        """
        if (self.verbose):print "query X is\n", X

        # Shape of Ypred_bag is [bags, X.shape[0]]
        # Shape of Y is [1, X.shape[0]], average over total bags
        Ypred_bag = np.empty((0, X.shape[0]))
        for i in range(0, self.bags):
            Y_entry = [self.learnersBag[i].query(X)] # convert a list [1,2,3]to [[1,2,3]]matrix
            Ypred_bag = np.append(Ypred_bag, Y_entry, axis=0) #append to make a matrix!

        Y = mode(Ypred_bag)[0][0]

        if (self.verbose):print "prediction bag Ypred_bag is\n", Ypred_bag
        if (self.verbose):print "prediction Y is\n",Y
        return Y

if __name__=="__main__":
    print "This is Bag leaner\n"

