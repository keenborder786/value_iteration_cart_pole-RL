# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:41:38 2020

@author: MMOHTASHIM
"""

import gym
import torch
import random
import torch.nn.functional as F
import numpy as np

## Making the evironment
env = gym.make("CartPole-v1")
env = env.unwrapped # to access the inner functionalities of the class

### agent
class agent():
  def __init__(self):
    self.action_space=[0,1]
    self.samples_obervation=[]
    self.model=torch.nn.Linear(4,1)
  def random_action(self):
    action=random.choice(self.action_space) 
    return action
  
  def sample_from_evnironment(self,sample_number):
    observation = env.reset()
    for _ in range(sample_number):
      action = self.random_action()
      observation, _, done, _ = env.step(action)
      self.samples_obervation.append(observation)
      if done:
        observation = env.reset()
     
    env.close()
  
  def generate_target(self,k,discount_factor):
    target=[]
    for sample in self.samples_obervation:
        observation = env.reset()
        action_q_value=[]
        q_a=[]
        for action in self.action_space:
            env.state = np.array(sample)
            for i in range(k):
              observation, reward, done, _ = env.step(action)
              with torch.no_grad():
                q_a.append(reward+discount_factor*(self.model(torch.tensor(observation,dtype=torch.float32))))##Bellman Equation
              if done:
                env.reset()
                env.state = np.array(sample)
            action_q_value.append(np.mean(q_a))
        target.append(np.max(action_q_value))
    return target
  def main_model_train(self,total_iteration,epochs,k,discount_factor,learning_rate,MODEL_PATH):
    
    loss_fn=F.mse_loss
    opt=torch.optim.SGD(self.model.parameters(),lr=learning_rate)
    current_loss=10*100
    i=0
    while True:
      
      prev_loss=current_loss
      
      target=self.generate_target(k,discount_factor)
      
      inputs=torch.tensor(self.samples_obervation,dtype=torch.float32)
      targets=torch.tensor(target,dtype=torch.float32).view(-1,)
      print('Iteration-Cycle--------[{}/{}]--Mean-Loss:{:.4f}'.format(i+1,total_iteration,current_loss))
      losses=[]
      
      for epoch in range(epochs):
        value_prediction=self.model(inputs)
        loss=loss_fn(value_prediction.view(-1,),targets)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
      
      current_loss=np.mean(losses)
      
      if current_loss>prev_loss or i>=total_iteration:    
        print('Saving the Model-----in current directory')
        torch.save(self.model,MODEL_PATH)
        break
      i+=1
    
        
  
