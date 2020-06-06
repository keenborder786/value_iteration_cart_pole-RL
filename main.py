# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:43:12 2020

@author: MMOHTASHIM
"""
from main_agent import agent
from agent_play import *
import params

def run(train=True,play=True):
    champ=agent()
    champ.sample_from_evnironment(params.SAMPLE_SIZE)
    if train:
        champ.main_model_train(total_iteration=params.TOTAL_ITERATION,epochs=params.EPOCHS,k=params.K,discount_factor=params.DISCOUNT_FACTOR,
                           learning_rate=params.LEARNING_RATE,MODEL_PATH=params.MODEL_PATH)
    if play:
        play_agent()
if __name__=='__main__':
    ## Testing_Time
    run(train=True,play=False)