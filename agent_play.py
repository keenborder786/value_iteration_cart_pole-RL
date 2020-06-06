# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:38:42 2020

@author: MMOHTASHIM
"""

import gym
import torch
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import seaborn  as sns
model=torch.load('champ')
model.eval()

def play_agent():
    action_space=[0,1]
    env = gym.make('CartPole-v0')
    env_2=gym.make('CartPole-v0')
    episode=[]
    rewards=[]
    for i_episode in range(200):
        env.reset()
        observation=env_2.reset()
        env_2 = env_2.unwrapped # to access the inner functionalities of the class
        rewards_epsiode=0
        for t in range(100):
            env.render()
            action_taken=[]
            for action in action_space:
                env_2.state = observation
                observation, reward, done_2, _ = env_2.step(action)
                inputs=torch.tensor(observation,dtype=torch.float32).view(1,-1)
                with torch.no_grad():
                    action_taken.append(model(inputs).detach().numpy().ravel()[0])
                if done_2:
                    env_2.reset()
                    env_2.state = observation
                    
            action=np.argmax(np.array(action_taken))
            observation, reward, done_1, info = env.step(action)
            rewards_epsiode+=reward
            if done_1:
                env.reset()
                print("Episode finished after {} timesteps".format(t+1))
                break
        rewards.append(rewards_epsiode)
        episode.append(i_episode)
    sns.lineplot(episode,rewards)
    plt.show()
    env.close()
    env_2.close()