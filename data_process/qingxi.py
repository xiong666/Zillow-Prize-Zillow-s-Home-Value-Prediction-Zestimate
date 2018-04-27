# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:34:50 2017

@author: admin
"""
import pandas as pd
import numpy as np
train_df =pd.read_csv('./data/train_2016_v2.csv')
test_df =pd.read_csv('./data/properties_2016.csv')
combine =[train_df,test_df]