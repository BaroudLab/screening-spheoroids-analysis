# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:27:02 2021

@author: Romain
"""

import pandas as pd
import seaborn as sns

from scipy.stats import mannwhitneyu as mann


"""Computes p-values for combinatorial-xp of FIC2 (simu/eto 1st/cis 1st)"""

FIC = [0.6728234512266468, 0.5774486289105668, 0.6198645848363363]+\
        [0.36409085344071873, 0.3905770914336029, 0.29995191720897224, 0.33075445046932694]+\
        [0.40563030297395036, 0.6198093632418007, 0.7101272801846579, 0.6488256124793462]
        
xp = ['Simultané'] * 3  + ['Eto premier'] * 4 + ['Cis premier'] * 4
            

FIC = pd.DataFrame({'FIC':FIC,'xp':xp})

sns.barplot(data=FIC,y='FIC',x='xp',palette='BuPu',ci='sd',capsize=.1)

print(mann(FIC[FIC.xp=='Eto premier'].FIC,FIC[FIC.xp=='Cis premier'].FIC,alternative='less'))
print(mann(FIC[FIC.xp=='Eto premier'].FIC,FIC[FIC.xp=='Simultané'].FIC,alternative='less'))
print(mann(FIC[FIC.xp=='Cis premier'].FIC,FIC[FIC.xp=='Simultané'].FIC))