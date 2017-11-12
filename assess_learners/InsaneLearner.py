import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.BagsOfLearners = []
        for i in range(0,20):
            self.BagsOfLearners = np.append(self.BagsOfLearners, bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def author(self):
        return 'yzhu319'
    def addEvidence(self,dataX,dataY):
        for bag in self.BagsOfLearners:
            bag.addEvidence(dataX,dataY)
    def query(self,X):
        self.predictionY = np.empty((0,X.shape[0]))
        for bag in self.BagsOfLearners:
            self.predictionY = np.append(self.predictionY, [bag.query(X)], axis=0)
        return np.mean(self.predictionY,axis = 0)
if __name__=="__main__":
    print "This is Insane leaner\n"