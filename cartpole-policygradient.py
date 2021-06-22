import gym
import numpy as np
import os
import time
import random

#trains in a few minutes and has almost maximum average score

#policy search using gradient descent

#learning rate and discount factor
#The reward for going out of bounds is the same as going in a hole

class policyGradient:
    #create nn that is just a linear layer    
    
    learning_rate= 0.000025
    gamma = 0.99
    batch_size = 10000
    max_t = 200
    
    #variables to be set based on environment
    env = None
    weights = None
    obsN = 0
    actionN = 0
    
    def __init__(self,environment):
        self.env = environment
        observation = self.env.reset()
        self.obsN = len(observation)
        self.actionN = env.action_space.n
        self.weights = np.random.rand(self.obsN,self.actionN)
        print("Initial Weights")
        print(self.weights)
        
    def policy(self,observation):
        action_prob = np.dot(observation,self.weights)
        exp = np.exp(action_prob)
        return exp/np.sum(exp)
        
    def nextAction(self,observation):
        action_prob = self.policy(observation)
        action = np.random.choice(list(range(self.actionN)), p=action_prob)
        return action
        
    def getWeights(self):
        return self.weights
    
    def findGradient(self,observation,action):
        P = self.policy(observation)      #the probability of doing each action
        #print("P",P)
        softMax = P.reshape(-1,1)
        #print("softMax",softMax)
        #jacobian of the probability to find gradient
        J = np.diagflat(softMax) - np.dot(softMax, softMax.T)
        #print("J",J)
        dSoftMax = J[action,:]
        #print("dSoftMax",dSoftMax)
        dLog = dSoftMax / P[action]
        #print("dLog",dLog)
        
        gradient = np.dot(observation.T[:,None], dLog[None,:])  #None adds the necessary dimentions
        #print("gradient",gradient)
        return gradient
        
    # a lot of help from this site: https://medium.com/samkirkiles/reinforce-policy-gradients-from-scratch-in-numpy-6a09ae0dfe12
    def solveWeights(self):
        for iteration in range(self.batch_size):
            
            observation = self.env.reset()
            
            #observations = []
            #actions = []
            
            gradients = []
            rewards = []
            
            ##future_returns = []
            
            
            if(iteration%1000==0):
                print(self.weights)
            
            for t in range(self.max_t):
                if(iteration%1000==0):
                    self.env.render()
                    
                action = self.nextAction(observation)
                gradient = self.findGradient(observation,action)
                
                #observations.append(observation)
                #actions.append(action)
                
                observation, reward, done, info = self.env.step(action)
                #each time add observation you started with, action took, reward got
                    
                rewards.append(reward)
                gradients.append(gradient)
                
                if(done):
                    ##for i in range(len(future_returns)-1,len(future_returns)+t):
                    ##    future_returns.append(np.sum(rewards[i:]))
                    break
            
            #find partial of log Pi(s,a)  use softmax policy
            #∇θlog(πθ(s,a))=ϕ(s,a)−Eπθ[ϕ(s,⋅)]
            #X=features=ϕ(s,a)
            #P=probabilities=πθ(s,a)
            #E[X]=X⋅P
            #print(self.weights)
            for i in range(len(rewards)):
                #gradient = self.findGradient(observations[i],actions[i])
    
                fr = sum([ r * (self.gamma ** r) for t,r in enumerate(rewards[i:])])
                self.weights += self.learning_rate * gradients[i] * fr 

    def runOnce(self,disp):
        score = 0
        observation = self.env.reset()
        for i in range(200):
            if(disp):
                env.render()
            
            action = rl.nextAction(observation)
            observation, reward, done, info = self.env.step(action)
            score+=reward
            
            if(done):
                break
        return score

env = gym.make('CartPole-v0')
rl = policyGradient(env)


rl.solveWeights()

        
        #time.sleep(0.1)
env.close()
env.render()

print("final weights\n",rl.getWeights())

score = 0
for i in range(100):
    score += rl.runOnce(disp=False)
print("score",score/100)

        
        
        