import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.BagsOfLearners = []
        for i in range(0,20):
            self.BagsOfLearners = np.append(self.BagsOfLearners, bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
        if(self.verbose): print "my Bags of Baglearner\n", self.BagsOfLearners
    def author(self):
        return 'yzhu319'
    def addEvidence(self,dataX,dataY):
        for bag in self.BagsOfLearners:
            bag.addEvidence(dataX,dataY)
    def query(self,X):
        self.predictionY = np.empty((0,X.shape[0]))
        if (self.verbose):print "in query, my Bags of Baglearner\n", self.BagsOfLearners
        for bag in self.BagsOfLearners:
            print "Hello ",bag
            print "bag.query", bag.query(X)
            self.predictionY = np.append(self.predictionY, [bag.query(X)], axis=0)
        if (self.verbose):print "prediction of 20bags\n", self.predictionY
        if (self.verbose):print "prediction of insane learner", np.mean(self.predictionY,axis = 0)
        return np.mean(self.predictionY,axis = 0)

if __name__=="__main__":
    print "This is Insane leaner\n"

