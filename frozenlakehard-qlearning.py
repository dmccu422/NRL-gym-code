import gym
import numpy as np
import os
import time
import random

#qlearning with 33% probability of wrong action happening
#passing is considered 0.78 average reward
#many things changed given the slippery condition
# rewarding fast completion didn't work
# learning rate had to be a bit higher

#exploration probability (e) being linearly decreasing didn't work. Some 1/x function found in an online solution was added to Q before argmax
# this is an important difference because I had all but that written myself and no version of slowly decaying e I came up with worked as good

class qlearning:
    learning_rate = 0.85
    discount_factor = 0.99
    e=0.9
    Q_s_a=np.array([])
    N_s_a=np.array([])
    
    def __init__(self):
        self.Q_s_a=np.zeros([16,4])
        print(self.Q_s_a)
        
    def nextAction(self,env,observation):
        if random.random()>self.e:   #explore probability is low
            action=np.argmax(self.Q_s_a[observation])
        else:       #explore probability is high
            action = env.action_space.sample()
        
        return action
        
    def updateQ(self,observation,expectedObs,action,reward):
        #target q-learning
        sample = reward + self.discount_factor * max(self.Q_s_a[expectedObs])
        #target SARSA
        #sample = reward + self.discount_factor * self.Q_s_a[expectedObs][action]
        
        self.Q_s_a[observation][action] = (1.0-self.learning_rate) * self.Q_s_a[observation][action] + self.learning_rate * sample
        #print("changed",observation,action," sample:",sample)
        #print(self.Q_s_a)
        
        if(self.e>0.05):
            self.e-=0.0001
    
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
        
env = gym.make('FrozenLake-v0',is_slippery=True)
rl = qlearning()
sumReward = 0
for i_episode in range(20001):
    observation = env.reset()
    rw = 0
    
    for t in range(200):
        #os.system('cls')
        #env.render()
        
        #action = rl.nextAction(env,observation)
        
        action = np.argmax( rl.getQ()[observation, :] + np.random.randn(1, env.action_space.n)*(1./(i_episode+1)) )
        #https://gym.openai.com/evaluations/eval_OAbMaV0TKe71Cq5Mtof7g/
       
        lastObs = observation
        observation, reward, done, info = env.step(action)
        rw=reward
        
        '''
        if(done and observation==15):
            reward = 1.0
        elif(done):
            reward = -1.0
        elif(observation==lastObs):
            reward = -1.0
        '''
        
        #print(action,observation, reward, done, info)
        
        rl.updateQ(lastObs,observation,action,reward)
        
        if done and reward!=1:
            print("Episode failed after {} timesteps".format(t+1))
            break
        elif done:
            print("Episode complete after {} timesteps".format(t+1))
            break
    
    
    sumReward+=rw
    if(i_episode%100==0):
        print('avg reward:',sumReward/100)
        sumReward=0
        
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

p = 0
for i in range(100):
    if(rl.testQ(env)):
        p+=1
print(p,"percent correct")



        
        
        
        