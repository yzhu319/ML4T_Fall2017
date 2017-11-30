"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def author(self):
        return 'yzhu319'

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma

        self.rar = rar
        self.radr = radr

        # initialize Q table as 2D np array filled with zeros
        self.Q_table = np.zeros((self.num_states, self.num_actions))
        self.dyna = dyna

        if self.dyna != 0:
            self.Tc = np.ndarray(shape=(num_states,num_actions,num_states)) #T[s,r,s']
            self.Tc.fill(0.0001)
            self.T = self.Tc/ np.sum(self.Tc, axis=2,keepdims = True)
            self.R = np.ndarray(shape=(num_states,num_actions))
            self.R.fill(-1)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if (rand.uniform(0.0,1.0) < self.rar):
            action = rand.randint(0, self.num_actions - 1)
        # otherwise, consult Q table at state self.s to choose action to take
        else:
            action = np.argmax(self.Q_table, axis=1)[self.s]

        #if self.verbose: print "s =", s,"a =",action
        #self.a = action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # given state-action pair (self.s, self.a), update Q_table at (self.s, self.a)
        # based on experience tuple (self.s, self.a, r, s_prime)

        self.Q_table[self.s][self.a] = (1- self.alpha)* self.Q_table[self.s][self.a] + \
            self.alpha * (r + self.gamma * self.Q_table[s_prime][np.argmax(self.Q_table, axis=1)[s_prime]])

        if (rand.uniform(0.0,1.0) < self.rar):
            action = rand.randint(0, self.num_actions - 1)
            self.rar = self.rar * self.radr
        # otherwise, consult Q table at state self.s to choose action to take
        else:
            # optimal for s_prime ***
            action = np.argmax(self.Q_table, axis=1)[s_prime]

        #if self.verbose: print "s_prime =", s_prime, "a =", action, "r =", r, "rar =", self.rar

        # Dyna
        if self.dyna !=0: #update T and R
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] +1
            self.T = self.Tc/ np.sum(self.Tc, axis=2, keepdims= True)
            self.R[self.s, self.a] = (1- self.alpha)*self.R[self.s, self.a] + self.alpha* r
            # iteration in dyna
            a_dyn_arr = np.random.randint(self.num_actions,size= self.dyna)
            s_dyn_arr = np.random.randint(self.num_states, size= self.dyna)
            for i in range(0, self.dyna):
                a_dyn = a_dyn_arr[i]
                s_dyn = s_dyn_arr[i]
                # infer s' from T
                #s_dyn_prime = np.random.choice(range(0, self.num_states), p = self.T[s_dyn, a_dyn, ])
                s_dyn_prime = np.random.multinomial(1, self.T[s_dyn,a_dyn,]).argmax()
                r_dyn = self.R[s_dyn][a_dyn]
                # update Q table inside loop
                self.Q_table[s_dyn][a_dyn] = (1- self.alpha)* self.Q_table[s_dyn][a_dyn] + \
                    self.alpha * (r_dyn + self.gamma* self.Q_table[s_dyn_prime][np.argmax(self.Q_table, axis=1)[s_dyn_prime]])


        self.s = s_prime
        self.a = action

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
