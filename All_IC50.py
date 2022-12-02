# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:11:24 2021

@author: Romain
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
""" Plots all the IC50 found in all xp into a boxplot"""
plt.rcParams['font.size'] = '14'
plt.rcParams['svg.fonttype']='none'


D1 = [12.7 , 10.24 , 18.45 , 11.17 , 18.76]
#12.7
D2 = [1.83 , 1.58 , 4.22  , 1.38 , 3.78]
#1.83
D3 = [0.71 , 1.04 ,  1.27 , 0.51 , 1.42]
#1.04
D4 = [ 0.85 ,  0.35 , 0.9]
#0.85
D5 = [ 0.83]
IC = D1+D2+D3+D4+D5
D = [1]*len(D1)+[2]*len(D2)+[3]*len(D3)+[4]*len(D4)+[5]
Data = pd.DataFrame({'ic':IC,'drug':['Cisplatine']*len(D),'day':D})

D1 = [11.3 , 1.76 , 8.11 , 13.18 , 16.31  ]
#11.3
D2 = [2.02 , 1.55 , 3.14 , 2.07 , 4.93  ]
#2.07
D3 = [0.93 , 0.87 , 1.48 , 0.49 , 0.97  ]
#0.93
D4 = [ 0.42 ,  0.26 , 0.61  ]
#0.42
D5 = [ 0.17   ]

IC = D1+D2+D3+D4+D5
D = [1]*len(D1)+[2]*len(D2)+[3]*len(D3)+[4]*len(D4)+[5]
Data2 = pd.DataFrame({'ic':IC,'drug':['Etoposide']*len(D),'day':D})

Data = Data.append(Data2)

sns.boxplot(data=Data,x='day',y='ic',hue='drug',palette='Set1')
plt.yscale('log')