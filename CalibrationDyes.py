# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:19:44 2021

@author: Romain

Code to analyse the calibration of CF647 and Cascade blue fluoerscence 
in function of concentration in 800µm chips.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from pandas import DataFrame, read_pickle
import matplotlib
from scipy.optimize import curve_fit

font = {'family' : 'TimeNewRoman',
        'size'   : 10}
matplotlib.rc('font', **font)

#%% Parameters


    
chanels = ['CY5','DAPI','FITC']
colors=  {'CY5':'red','DAPI':'blue','FITC':'green'}
concentrations = ['5nM','13nM','35nM','96nM','258nM','694nM','1863nM','5000nM']
Concentrations = [5,13,35,96,258,694,1863,5000]
master_path=r'\\gaia.pasteur.fr\Multicell\Romain\data\E7\\'
save_path = r'E7\\'

columns = ['Well','Dye','Concentration','DAPI','FITC','CY5','isGood']



#Exposure time depending on the dye used and channel (comments below)
def expo(dye):
    
    if dye=='DAPI':Expo = [[300,300,300,300,300,300,300,300], #Cy5
            [300,300,300,300,300,300,100,50],       # Dapi
            [300,300,300,300,300,300,300,300]]       #Fitc
    
    else:Expo = [[300,300,300,300,300,300,100,50],        #Cy5
            [300,300,300,300,300,300,100,100],       # Dapi
            [100,100,100,100,100,100,100,100]]       #Fitc

    return dict(zip(chanels,[dict(zip(concentrations,expo)) for expo in Expo]))




#%% Functions for reading wells fluo/concentration

# Reading fluo in center (using circ. mask, radius half of the well's radius)
# without background noise, estimated in corners (mask_noise)
# Returns float

def read_picture(img):
    h,w = np.shape(img)
    center = (w//2,h//2)      
    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= h//4
    
    summed = X/80+Y/140
    mask_noise = (summed < 2) + (summed > 19)
    

    return np.ma.median(np.ma.array(img,mask=~mask))-np.ma.median(np.ma.array(img,mask=~mask_noise))


# For one concentration and one channel, reads all wells and returns
# fluo values in list

def read_chanel(path):
    res = []
    
    for im in tqdm(os.listdir(path)):
        if im.endswith('.tif'):           
            res.append(read_picture(cv2.imread(path+im,-1))) 
       
    return res




# For  one concentration, reads all channels with previous func, and return a
# DataFrame looking like:
    
# 'Well'     'Dye'               'Concentration'     'DAPI'     'FITC'     'CY5'          'isGood'
#-----------------------------------------------------------------------------------------------------------
#  well1    the dye in well    its concentration   fluo in dapi/fitc/cy5 measured   if value is not aberent
#  well2      .........   
    
def read_concentration(path,dye,c):
    Chip = DataFrame(columns=columns)
    
    for chan in chanels:
        Chip[chan] = read_chanel(path+chan+'\\')
        
    Chip.Well = [i+1 for i in range(len(Chip))]    
    Chip.Dye = [dye]*len(Chip)
    Chip.Concentration = [c]*len(Chip)
    Chip.isGood = Chip[dye]>=np.median(Chip[dye])*.8
    return Chip
        

# For all concentrations, reads the previous DF and concatenates them
    
def read_all_concentrations(dye):    
    Chip = DataFrame(columns=columns)
    
    for c in concentrations:
        print(f'__Reading concentration {c}__')
        Chip = Chip.append(read_concentration(master_path+dye+'_'+c+'\\',dye,c))
    
    return Chip



        
#%% Setup plot functions

# Returns te data to be plot by channel
# if dye is CY5 and chan not CY5, missing values for 5nM,13nM and 35nM.

def get_data(chan):
    
    if dye=='CY5' and chan!=dye:a = [Wells[(Wells.Concentration==c)&(Wells.isGood)&\
                                           (Wells.Concentration!='5nM')\
                                           &(Wells.Concentration!='13nM')\
                                           &(Wells.Concentration!='35nM')\
                                           &(Wells[chan]>0)]\
                                            [chan]/exposure[chan][c] for c in concentrations]
        
        
    else:a = [Wells[(Wells.Concentration==c)&(Wells.isGood)]\
                                              [chan]/exposure[chan][c] for c in concentrations]
     
    
    X,Y = [],[]
    for y,c in zip(a,Concentrations):
            Y.extend(y)
            X.extend([c]*len(y))
            
    return a,X,Y


# Function for fit: f(c) = Ac^B+D (here it is loglog-scale)

def f(x,a,b,k):
    
    return np.log10(a*10**(k*x)+b)


#%% plot 


def plot(D):
    
    global Wells,exposure,dye
    dye=D
    exposure = expo(dye)
    
    #If data already read, it is stored in pkl file
    
    try:
        Wells = read_pickle(save_path+dye+".pkl")
        for c in concentrations:
            if len(Wells[(Wells.Concentration==c)])==0:
                temp = read_concentration(master_path+dye+'_'+c+'\\',dye,c)
                Wells = Wells.append(temp)
                del temp
    
    
    # if not existing or incomplete, read and write
    
    except (FileNotFoundError,KeyError):
        Wells = read_all_concentrations(dye)
        
    Wells.to_pickle(save_path+dye+".pkl")
                
    
    
    
    #plot boxplot (adaptative width managed with lambda bc log/log scale)
    
    
    
    fig,ax = plt.subplots(figsize=(7,7))
    
    w=.1 #log width of boxplot
    width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.) #left/right bounds of boxplot
    
    
    
    
    
    for chan in chanels:
        
        #plot data in boxplot
        a,X,Y = get_data(chan)
        ax.boxplot(a,positions=Concentrations,widths=width(Concentrations,w))
        
        
        #scatter plot for individual points
        for y,C,c in zip(a,Concentrations,concentrations):
            kargs = {'alpha':.1,'color':colors[chan]}
            if C==Concentrations[0]:kargs['label']=chan        
            ax.scatter(np.random.normal(1,.04,len(y))*C,y,**kargs)
        
        
        #Doing fit and ploting it on dye data
        
        
        (a,b,k),Cov = curve_fit(f, np.log10(X), np.log10(Y), bounds=([0,0,0],[1,20,1]))
        
        
        x = np.logspace(np.log10(X[0]*.4),np.log10(X[-1]/.9),100)
        y = 10**f(np.log10(x),a,b,k)
        ax.plot(x,y,colors[chan],label=f"I={a:.2e} c^{k:.2e}+{b:.2e}",alpha=.7)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if dye=='DAPI':ax.set_xlabel('Cascade Blue concentration')
    else:ax.set_xlabel('CF647 concentration')
    ax.set_ylabel('Median signal in center of drop, by unit of time exposure')
    
    ax.legend()

 
#%% Main loop

# Function that will be called from other codes to read read barcoding
# Returns fit parameters

def calibration(D):
    global Wells,exposure,dye
    dye=D
    exposure = expo(dye)
    
    #If data already read, it is stored in pkl file
    
    try:
        Wells = read_pickle(save_path+dye+".pkl")
        for c in concentrations:
            if len(Wells[(Wells.Concentration==c)])==0:
                temp = read_concentration(master_path+dye+'_'+c+'\\',dye,c)
                Wells = Wells.append(temp)
                del temp
    
    
    # if not existing or incomplete, read and write
    
    except (FileNotFoundError,KeyError):
        Wells = read_all_concentrations(dye)
        
    Wells.to_pickle(save_path+dye+".pkl")
    
    
    
    a,X,Y = get_data(dye)
    
    a,b = np.polyfit(X,Y,1)
    (a,b,k),Cov = curve_fit(f, np.log10(X), np.log10(Y), bounds=([0,0,0],[1,20,1]))
    
    return a,b,k
    


# If code is executed from here, plot the two calibrations.
if __name__=='__main__':
    
    plot('CY5')
    plot('DAPI')