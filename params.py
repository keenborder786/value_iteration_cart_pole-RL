# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:44:40 2020

@author: MMOHTASHIM
"""
import os

##Hyperparameters:
TOTAL_ITERATION=100000
EPOCHS=15
K=15##K IS THE NUMBER OF SAMPLE TAKEN TO CALCULATE THE Q_A THAT IS THE EXPECTED FUTURE VALUE
DISCOUNT_FACTOR=0.1
LEARNING_RATE=1e-5
MODEL_PATH=os.path.join(os.getcwd(),'champ')
SAMPLE_SIZE=50#Number of Examples our agents trains on 