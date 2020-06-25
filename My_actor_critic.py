import tensorflow as tf
from tensorflow import keras
import gym
#import atari_py
from collections import deque
import numpy as np
import random
random.seed(0)

env = gym.make('Boxing-v0') #boxing
#env = gym.make('SpaceInvaders-v0')
#env = gym.make('MinitaurBulletEnv-v0')

D = deque()
env  = env
    #sess = sess
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = .995
gamma = .95
tau   = .125
observetime = 50000
mb_size = 50
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False

def build_actor_network():
    model = keras.Sequential()
    model.add(keras.layers.Input(env.observation_space.shape))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
        
def build_critic_network():
    model = keras.Sequential()
    model = keras.layers.Input(env.observation_space.shape, env.action_space.shape)
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.compile(optimizer='adam', loss='mse')

for t in range(observetime):
    AQ = build_actor_network #Q stands for quality
    proposed_action = np.argmax(AQ) #action with highest q value
    CQ = build_critic_network
    approved_action = np.argmax(CQ, proposed_action)
    observation, reward, done, info = env.step(approved_action)

    
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()    
    '''                # Uncomment to see game running 
    AQ = build_actor_network.predict(env.action_space.n) #Q stands for quality
    proposed_action = np.argmax(AQ) #action with highest q value
    CQ = build_critic_network.predict(proposed_action)
    approved_action = np.argmax(CQ)
    observation, reward, done, info = env.step(approved_action)'''
    #Q = build_actor_network.predict(env.action_space.n) #Q stands for quality
    #action = np.argmax(Q) #action with highest q value
    #action = np.random.randint(0, env.action_space.n, size=1)[0]
    #observation, reward, done, info = env.step(action)'''
    #approved_action = np.random.randint(0, env.action_space.n, size=1)[0]
    action = np.random.randint(approved_action, env.action_space.n, size=1)[0]
    observation, reward, done, info = env.step(action)

env.close()    
#print('Game ended! Total reward: {}'.format(reward))