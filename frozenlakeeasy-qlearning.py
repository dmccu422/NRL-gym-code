import gym
import numpy as np
import os
import time
import random

#solves quickly with no randomness in the next state from an action

#qlearning with exploration function (N_q_a)
#the next state's q values are increased in directions that haven't been explored much

#exploration probability (e) is decreased gradually

#learning rate and discount factor tuned for stability with very few episodes.
#The reward for going out of bounds is the same as going in a hole

class qlearning:
    learning_rate = 0.7
    discount_factor = 0.99
    e=0.9
    Q_s_a=np.array([])
    N_s_a=np.array([])
    
    def __init__(self):
        self.Q_s_a=np.zeros((16,4))
        self.N_s_a=np.ones((16,4))  #visit count
        print(self.Q_s_a)
        
    def nextAction(self,env,observation):
        if random.random()>self.e:   #explore probability is low
            action=np.argmax(self.Q_s_a[observation])
        else:       #explore probability is high
            action = env.action_space.sample()
        
        return action
        
    def updateQ(self,observation,expectedObs,action,reward):
        sample = reward + self.discount_factor * max(self.Q_s_a[expectedObs] + .1 / self.N_s_a[expectedObs])
        self.N_s_a[observation][action]+=1
        
        self.Q_s_a[observation][action] = (1.0-self.learning_rate) * self.Q_s_a[observation][action] + self.learning_rate * sample
        #print("changed",observation,action," sample:",sample)
        #print(self.Q_s_a)
        
        if(self.e>0.05):
            self.e-=0.01
    
    def getQ(self):
        return self.Q_s_a
        
    def testQ(self,env):
        good=False
        observation = env.reset()
        for i in range(16):
            action=np.argmax(self.Q_s_a[observation])
            observation, reward, done, info = env.step(action)
            if(done and observation==15):
                good=True
        return good
        
        
env = gym.make('FrozenLake-v0',is_slippery=False)
rl = qlearning()

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        #os.system('cls')
        #env.render()
        
        #print(observation)
        action = rl.nextAction(env,observation)
        lastObs = observation
        observation, reward, done, info = env.step(action)
        
        if(done and observation==15):
            reward = 1.0
        elif(done):
            reward = -1.0
        elif(observation==lastObs):
            reward = -1.0
        
        #print(action,observation, reward, done, info)
        
        rl.updateQ(lastObs,observation,action,reward)
        
        if done and reward==-1:
            print("Episode failed after {} timesteps".format(t+1))
            break
        elif done:
            print("Episode complete after {} timesteps".format(t+1))
            break
            
        
        #time.sleep(0.1)
env.close()
env.render()

Q = rl.getQ()
print(Q)
for i in range(16):
    action = np.argmax(Q[i])
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

print(rl.testQ(env))



        
        
        
        