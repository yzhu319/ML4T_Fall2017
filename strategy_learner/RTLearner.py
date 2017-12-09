"""
@Author: Yuanzheng Zhu (yzhu319)
Random Decision Tree leaner .
Convert from a regression learner to a classification learner
Previous, leaf value = xxx.mean() -->  mode(xxx)[0][0]   //from scipy.stats import mode
"""

import numpy as np
from scipy.stats import mode

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'yzhu319'

    def build_tree(self, data):
        dataX = data[:, 0:-1]
        dataY = data[:, -1]
        dataY = dataY.reshape((dataX.shape[0], 1)) #convert to column vector

        #print "data is\n", data
        if (data.shape[0] <= self.leaf_size) or (np.unique(data[:,-1]).size == 1):
            new_leaf = np.array([[-99,  mode(data[:,-1])[0][0] , np.nan, np.nan]])
            return new_leaf
        else:
            #randomly choose a column a dataX, a factor
            factor = np.random.randint(0, dataX.shape[1])
            splitVal = np.median(data[:,factor])

            #left branch of sub-tree
            left_tree_data = data[data[:,factor]<=splitVal]
            #print "left tree data\n", left_tree_data
            if left_tree_data.shape[0] == data.shape[0]:
                return np.array([[-99, mode(data[:,-1])[0][0] , np.nan, np.nan]])
            elif left_tree_data.shape[0] <= self.leaf_size:
                left_tree = np.array([[-99, mode(left_tree_data[:,-1])[0][0], np.nan, np.nan]])
            else:
                left_tree = self.build_tree(left_tree_data)
            #print "left tree\n", left_tree

            #right branch of sub-tree
            right_tree_data = data[data[:,factor]>splitVal]
            #print "right tree data\n", right_tree_data
            if right_tree_data.shape[0] == data.shape[0]:
                return np.array([[-99, mode(data[:,-1])[0][0], np.nan, np.nan]])
            elif right_tree_data.shape[0] <= self.leaf_size:
                right_tree = np.array([[-99, mode(right_tree_data[:,-1])[0][0], np.nan, np.nan]])
            else:
                right_tree = self.build_tree(right_tree_data)
            #print "right tree\n", right_tree

            new_root = np.array([[factor, splitVal, 1, left_tree.shape[0]+1]])
            #print "root\n", new_root
            return np.append(new_root,np.append(left_tree,right_tree, axis=0), axis=0)


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to DTlearner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # data used to build the tree; append dataY to dataX, N+1 column
        dataY = dataY.reshape((dataX.shape[0],1))
        data = np.concatenate((dataX, dataY), axis = 1) #concatenate horizontally

        if(self.verbose):print "original data is\n",data
        # build and save the DT model
        tree = self.build_tree(data)
        self.tree = tree
        if(self.verbose):print "tree is \n",tree

        #return tree

    def query(self,X):
        """
        @summary: query a RT .
        @param X: should be a numpy array with each row corresponding to a specific query.
        @returns the predicted value Y from RTlearner.
        """

        Y = []
        # X is matrix of [num_enties, num_features]
        # loop through rows, x_entry is row vector
        if (self.verbose):print "query X is\n", X
        for x_entry in X:
            row = 0
            feature = int(self.tree[row,0])
            while (feature >= 0): # if it is not a leaf node, enter loop
                splitVal = self.tree[row,1]
                left = int(self.tree[row,2])
                right = int(self.tree[row,3])
                if(x_entry[feature] <= splitVal):
                    row = row + left
                else:
                    row = row + right
                feature = int(self.tree[row,0])
            Y.append(self.tree[row,1])

        if (self.verbose):print "prediction Y is\n",Y
        return Y

if __name__=="__main__":
    print "This is RT leaner\n"

