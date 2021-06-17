import gym
import numpy as np
import random
env = gym.make('CartPole-v0')

#Ran for 30 minutes and achieved max score of 199 several times with average of 100 after 450000 iterations.
# Oddly the average is ~130 after ~5 minutes (70000 iterations) and goes down from there, overfitting?

#q learning with rounded observation space
#exploration probability (e) is fixed
#learning rate and discount factor are default

#could be further optimized but this is good enough to show that it works. Other methods will be my priority

learning_rate = 0.5
discount_factor = 1
e=.1
Q_s_a=np.random.rand(100,100,100,100,2)

#round the observation space and map it to an integer that will be within the range [0,99]
def simplify_observation(observation):
    #print(observation)
    observation[0] /= 1
    observation[3] /= 2
    observation[1] /= 1
    observation[2] /= 2
    return tuple([int(x) for x in np.round(observation,1)*10])

sum=0
for iteration in range(500000):
    if(iteration%10000==0):
        print(iteration)

    #reset the game state
    observation = env.reset()
    obs_simp = simplify_observation(observation)

    for score in range(500):    #200 is the max score
        if(iteration%10000==0):
            env.render()
        
        #act based on Q_s_a or explore randomly based on exploration prob
        if random.random()>e:   #explore probability is low
            action=np.argmax(Q_s_a[obs_simp])
        else:       #explore probability is high
            action = env.action_space.sample()
            
        lastOb = obs_simp
        observation, reward, done, info = env.step(action)
        #simplify observation space
        obs_simp = simplify_observation(observation)
        #observation format: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        
        if(done and score<199):   #its bad to fail
            reward = -1
        
        #q-learning
        sample = reward + discount_factor * max(Q_s_a[obs_simp])
        
        Q_s_a[lastOb][action] = (1.0-learning_rate) * Q_s_a[lastOb][action] + learning_rate * sample
            
        #calculate average score per 10000 iterations and print
        if(done):
            sum+=score
        if(done and iteration%10000==0):
            print('score:',score)
            print('average:',sum/10000)
            sum=0
        if(done):
            break

    env.close()

