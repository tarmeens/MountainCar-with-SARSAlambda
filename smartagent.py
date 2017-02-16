import sys

import pylab as plb
import numpy as np
import mountaincar

class SmartAgent():
    """Here's my job!
    """

    def __init__(self, mc = None, parameter1 = 3.0):
        
        if mc is None:
            self.mc = mountaincar.MountainCar()
        else:
            self.mc = mc

        self.parameter1 = parameter1
        
    def gauss(self,state, center, width):
        """ apply gaussian formula to i-state and j-center
            @param3 width: distance between centers
        """
        return np.exp( - (((center[0] - state[0])**2)/(width[0]**2)) - (((center[1] - state[1])**2)/(width[1]**2)))

    def input_response(self, state, centers, width):
        """ compute input response for each center
        """
        size = centers.shape[0] * centers.shape[1]
        # response of the input layer
        out = np.zeros(size) 
        for i in range(0,size):
            row = int(i / centers.shape[0])
            col = i % centers.shape[0]
            out[i] = self.gauss(state, centers[row,col], width)
        return out

    def soft_max(self, q, tau):
        """ returns probabilities given neuron response
            @param2 tau: exploration temperature parameter
        """
        denom = np.sum(np.exp(q/tau), axis = 0)
        prob = np.zeros(q.shape[0])
        for i in range(0, q.shape[0]):
            prob[i] = np.exp(q[i]/tau)/denom
        return prob

    def visualize_trial(self, id_, eta = 0.01, lambda_ = 0.8, tau = 0.05, tau_decay = False, init = 1,  n_learn = 100, show = True,  n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # layer size of the network
        size = 10
        # "fixed" parameters
        gamma = 0.95
        # keep track of current state
        curr_state = np.zeros(2)
        prev_state = np.zeros(2)
        center_dist = np.zeros(2)
        # escape latency
        self.latency_list = np.zeros(n_learn)
        #init network
        centers = np.zeros((size,size,2))
        # init centers
        for i in range(size):
            for j in range(size):
                #centers[i,j,0] = (180 * (i* size + j + 1)/(size*size + 1)) - 150 # position center
                #centers[i,j,1] = (30 * (i* size + j + 1)/(size*size + 1)) - 15 #speed center
                #centers[i,j,0] = 180 * ((j + 1) / (size + 1)) - 150 # position center
                #centers[i,j,1] = 15 - 30* ((i + 1) / (size + 1))  #speed center
                centers[i,j,0] = 180 * (j / (size - 1)) - 150 # position center
                centers[i,j,1] = 30* ( i / (size - 1)) - 15 #speed center
        center_dist[0] = 180/(size - 1)
        center_dist[1] = 30/(size - 1)
        # weights in-out layer
        if init == 0:
            print("Agent " + str(id_) + ": weights init to 0")
            weights = np.zeros((3,size*size))  #init to 0
        elif init == 1:
            print("Agent " + str(id_) + ": weights init to 1")
            weights = np.ones((3,size*size))  #init to 1
        else:
            print("Agent " + str(id_) + ": wrong initialization number!")
            
        for n in range(n_learn):
            # trace associated to weights
            trace = np.zeros((3,size*size)) 
            # make sure the mountain-car is reset
            self.mc.reset()
            #save current state
            prev_state[0] = self.mc.x
            prev_state[1] = self.mc.x_d
            #compute first q
            prev_q = weights.dot(self.input_response(prev_state,centers,center_dist))
            #compute probabilities
            prob = self.soft_max(prev_q, tau)
            #take next action according to probabalities
            prev_action = np.random.choice(3, p = prob)
            # prima azione
            self.mc.apply_force(prev_action - 1)
            self.mc.simulate_timesteps(100, 0.01)
            while True:
                #actions according to current state
                curr_state[0] = self.mc.x
                curr_state[1] = self.mc.x_d
                # tau decay
                if tau_decay:
                    tau = 1 * np.exp(-n*0.1) + 0.005
                #neuronal output
                current_q = weights.dot(self.input_response(curr_state,centers,center_dist))
                #compute probabilities
                prob = self.soft_max(current_q, tau)
                #take next action according to probabalities
                action = np.random.choice(3,p = prob)
                #compute errors
                error = self.mc.R - (prev_q[prev_action] - gamma*current_q[action])
                # SWAP NEXT TWO LINES?
                #update every element
                trace = trace * gamma * lambda_
                # update row of taken action
                trace[prev_action] = trace[prev_action] + self.input_response(prev_state, centers, center_dist)
                #update weights
                weights =  weights + eta*error*trace
                
                #update previous parameters
                prev_q = current_q
                prev_action = action
                prev_state = curr_state
                # save time
                self.latency_list[n] = str(self.mc.t)
                #save weights
                self.W = weights
                # check for rewards
                if self.mc.R > 0.0:
                    print("\r Agent " + str(id_) + ": reward obtained at t = " + str(self.mc.t) + ", i = " + str(n), np.sum(weights))
                    break            
                
                #control car
                self.mc.apply_force(action - 1)
                # simulate the timestep
                self.mc.simulate_timesteps(100, 0.01)
       
        # prepare for the visualization
        if show:
            plb.ion()
            plb.pause(0.0001)
            mv = mountaincar.MountainCarViewer(self.mc)
            mv.create_figure(n_learn, n_learn)
            plb.draw()
        
            # VISUALIZE RESULTS
            while 1:
                print('\rt = ' + str(self.mc.t))
                sys.stdout.flush()

                #actions according to current state
                curr_state[0] = self.mc.x
                curr_state[1] = self.mc.x_d
                #neuronal output
                current_q = weights.dot(self.input_response(curr_state,centers,center_dist))
                #compute probabilities
                prob = self.soft_max(current_q, tau)
                #take next action according to probabalities
                action = np.random.choice(3,p = prob)
                #control car
                self.mc.apply_force(action - 1)
                # simulate the timestep
                self.mc.simulate_timesteps(100, 0.01)
                # update the visualization
                mv.update_figure()
                plb.draw()   
                plb.pause(0.01)
                # check for rewards
                if self.mc.R > 0.0:
                    print("\r Agent " + str(id_) + ": reward obtained at t = " + str(self.mc.t))
                    break  
            plb.show(block = True) # leave the window open at the end of the simulation 
    

if __name__ == "__main__":
    s = SmartAgent()
    s.visualize_trial(1)
        

