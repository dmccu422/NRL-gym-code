import gym
import numpy as np
import random
env = gym.make('CartPole-v0')

#approximate Q learning using value functions
#train a linear model as a function of the observations

#about 27 average after several training iterations. Discount factor is best low meaning the value function isn't as useful as reward is

#function variables could be expanded even more
#bulding point for neural network (DRL) where multiple layers are used instead of just linear approximate

weights = np.random.rand(4+ 16) * 2 - 1 #range of [-1, 1]
learning_rate = 0.5
discount_factor = 0.3
e=0.05

sum=0
for iteration in range(100000):
    if(iteration%1000==0):
        print(iteration)

    observation = env.reset()
    observation = np.append(observation, np.reshape((np.reshape(observation,(4,1))*np.reshape(observation,(1,4))),(16,1)))
    Q_s_a = np.dot(weights,observation)

    for score in range(500):
        #env.render()
        #act based on Q_s_a or explore randomly based on exploration prob
        if random.random()>e:   #explore probability is low
            action = int(Q_s_a>0)
        else:       #explore probability is high
            action = env.action_space.sample()
        #take the chosen action
        observation, reward, done, info = env.step(action)
        #observation format: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        
        #add second order of observations
        observation = np.append(observation, np.reshape((np.reshape(observation,(4,1))*np.reshape(observation,(1,4))),(16,1)))
        
        if(done and score<199):
            reward = -1
        
        #linear value functions
        Q_s_a_next = np.dot(weights,observation)
        
        difference = reward + discount_factor * Q_s_a_next - Q_s_a
        weights = weights + learning_rate * difference * observation
        #normalize the weights so they don't grow bigger with each iteration
        weights = weights/max(abs(weights))
        Q_s_a = Q_s_a_next
        
        #calculate average score per 10000 iterations and print
        if(done):
            sum+=score
        if(done and iteration%1000==0):
            print('score:',score)
            print('average:',sum/1000)
            sum=0
        if(done):
            break

    env.close()

print(weights)



