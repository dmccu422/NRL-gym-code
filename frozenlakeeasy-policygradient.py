import gym
import numpy as np
import os
import time
import random

#code attempted on my own is commented out

#policy search using gradient descent

#learning rate and discount factor are pretty standard
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
        self.obsN = 1#len(observation)
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
        action = np.random.choice(list(range(self.actionN)), p=action_prob[0])
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
        dLog = dSoftMax / P[0,action]
        #print("dLog",dLog)
        
        gradient = np.dot(observation, dLog[None,:])  #None adds the necessary dimentions
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
        
        
'''
class policyGradient:
    #create nn that is just a linear layer
    #2D input 4D output
    #         - left
    #       / /
    #x pos -  - down
    #       x x   
    #y pos -  - right
    #       \ \
    #         - up
    
    observations = []
    actions = []
    rewards = []
    future_returns = []
    num_episodes = 0
    weights = np.random.rand(2,4)
    learning_rate= 0.5
    batch_size = 1
    max_t = 100
    
    def __init__(self):
        print(self.weights)
        
    def obsToGrid(self,observation):
        return np.array([[int(observation/4)+1,observation%4+1]])
        
    def policy(self,observation):
        obsgrid = self.obsToGrid(observation)
        #print(observation,obsgrid)
        action_prob = np.dot(obsgrid, self.weights)
        exp = np.exp(action_prob)
        return exp/np.sum(exp)
        
    def nextAction(self,env,observation):
            
        action_prob = self.policy(observation)
        if(np.sum(action_prob)!=0):
            action_prob = action_prob+np.min(action_prob)
            action_prob = action_prob/(np.sum(action_prob))
        action = np.random.choice([0,1,2,3], p=action_prob[0].tolist())
        return action
        
    def getWeights(self):
        return self.weights
        
        
        loss = -sum(log prob * future returns) / number of episodes
        #change the weights by the loss
        weights = weights + learning_rate * loss
        #normalize the weights so they don't grow bigger with each iteration
        weights = weights/max(abs(weights))
        
    
    def runBatch(self,env):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.future_returns=[]
        
        for iteration in range(self.batch_size):
            observation = env.reset()
            for t in range(self.max_t):
                action = self.nextAction(env,observation)
                
                self.observations.append(observation)
                self.actions.append(action)
                
                observation, reward, done, info = env.step(action)
                #each time add observation you started with, action took, reward got
                
                if(done):
                    reward=-1.0
                self.rewards.append(reward)
                
                if(done):
                    for i in range(len(self.future_returns)-1,len(self.future_returns)+t):
                        self.future_returns.append(np.sum(self.rewards[i:]))
                    self.num_episodes+=1
                    break
            
        #find partial of log Pi(s,a)  use softmax policy
        #∇θlog(πθ(s,a))=ϕ(s,a)−Eπθ[ϕ(s,⋅)]
        #X=features=ϕ(s,a)
        #P=probabilities=πθ(s,a)
        #E[X]=X⋅P
        print(self.weights)
        for i in range(len(self.observations)):
            P = self.policy(self.observations[i])      #the probability of doing each action
            #print("P",P)
            softMax = P.reshape(-1,1)
            #print("softMax",softMax)
            #jacobian of the probability to find gradient
            J = np.diagflat(softMax) - np.dot(softMax, softMax.T)
            #print("J",J)
            dSoftMax = J[self.actions[i],:]
            #print("dSoftMax",dSoftMax)
            dLog = dSoftMax / P[0,self.actions[i]]
            #print("dLog",dLog)
            
            obsgrid = self.obsToGrid(observation)
            gradient = np.dot(obsgrid.T, dLog[None,:])
            #print("gradient",gradient)
            
            self.weights += self.learning_rate * self.future_returns[i] * gradient
'''
        
env = gym.make('FrozenLake-v0',is_slippery=False)
rl = policyGradient(env)

rl.solveWeights()

env.close()

print("final weights\n",rl.getWeights())

score = 0
for i in range(100):
    score += rl.runOnce(disp=False)
print("score",score/100)




Q = rl.getWeights()
print(Q)
for i in range(16):
    action_prob = rl.policy(i)
    action = np.argmax(action_prob[0])
    
    if(i%4==0):
        print()
    if(action==0):
        print("\u2190",end='')
    elif(action==1):
        print("\u2193",end='')
    elif(action==2):
        print("\u2192",end='')
    elif(action==3):
        print("\u2191",end='')
print()




        
        
        
        