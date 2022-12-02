# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:04:32 2021

@author: Romain
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd

os.chdir(r'\\gaia.pasteur.fr\Multicell\Romain\analysis\E4')

Chips = pd.read_pickle('Plates.pkl')

dates = Chips.date.unique()
dates.sort()

Chips.date = np.array(Chips.date,dtype=int)-int(dates[0])+1
Chips = Chips[Chips.row=='F']

#%%





def show():
    global Chips
    fig,axes = plt.subplots(1,2)
    ax,bx = axes
    
    print(Chips.groupby(['date']).count().row)
    kargs = {'data':Chips,'x':'date','y':'Radius','ax':ax,'order':range(1,len(dates)+3)}
    # sns.stripplot(**kargs,alpha=.6,color='orange')
    # sns.violinplot(**kargs,palette='viridis')
    sns.boxplot(**kargs,color='gray')
    sns.stripplot(**kargs,palette='viridis',alpha=.5)



    
    ax.set_xlabel('Days after seeding')
    ax.set_ylabel('Spheroids Radius (um)')
    
    
    kargs = {'data':Chips,'x':'date','y':'Viability','ax':bx,'order':range(1,len(dates)+3)}
    sns.boxplot(**kargs,color='gray')
    sns.stripplot(**kargs,palette='viridis',alpha=.5)



    bx.set_xlabel('Days after seeding')
    bx.set_ylabel('Spheroids Viability')
    bx.set_ylim((-.1,1.1))
    
    return Chips



show()
