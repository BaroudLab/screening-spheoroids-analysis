# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:04:48 2021

@author: Romain


THIS CODE IS THE SAME AS GETDRUGS.PY BUT WORKS WITH ONLY ONE DRUG.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

from Params import Dyes_Drugs,BarCoding,Exposition,data_path
from CalibrationDyes import calibration

#%%


columns = ['chip','well','control']
columns2 = ['chip','well','dapi','fitc','cy5']
Calib = {'DAPI':calibration('DAPI'),'CY5':calibration('CY5')}

def dye_to_drug(I,dye,expo):
    
    """
    Parameters
    ----------
    I : intensity of fluorescence.\n
    dye : florescnet dye used.\n
    expo : exposition time used for imaging

    Returns
    -------
    Drug based on barcoding data.\n
    Concentration of the drug based on calibration data of the dye, and 
    barcoding data (ratio drug to dye).    
    """

    
    a,b,k = Calib[dye]
    
    return ((I/expo-b)/a)**(1/k)*BarCoding[Dyes_Drugs[dye]]*1e-3
    
    
   
    
    
    
def controls_thresholds(Chip):
    
    n = len(BarCoding['Control'])
    
    mix = GaussianMixture(n).fit(np.array(Chip.fitc).reshape(-1,1))
    mus = mix.means_.flatten()
    
    return (mus[1:]+mus[:-1])/2
    


def dye_to_control(Chip):
    
    """
    For a n+1-gaussian repartition (Not control + n controls), finds :\n
        0 for not control\n
        1 for lowest concentration
        2 for mid concentration
        3 ...
    Returns what control is in which well for all the chip
    """
    
    thresh = controls_thresholds(Chip)
    fitc = Chip.fitc
    I = np.sum([fitc>=t for t in thresh],0)
    res = []
    for i in I:
        res.append(BarCoding['Control'][i])
    return res

#%%


def read_well(path,im_name):
    """
    Parameters
    ----------
    path : path to Barcoding diretory.\n
    im_name : name of DAPI image.

    Returns
    -------
    DAPI_lvl
    FITC_lvl
    CY5_lvl
    """
    
    
    DAPI = cv2.imread('\\'.join((path,'DAPI',im_name)),-1)
    CY5 = cv2.imread('\\'.join((path,'CY5',im_name.replace('DAPI','CY5'))),-1)
    FITC = cv2.imread('\\'.join((path,'FITC',im_name.replace('DAPI','FITC'))),-1)
    
    h,w = np.shape(CY5)
    center = (w//2,h//2)      
    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= h//4
    
    summed = X/80+Y/140
    mask_bg = (summed < 2) + (summed > 19)
    

    DAPI_bg = np.ma.median(np.ma.array(DAPI,mask=~mask_bg))
    DAPI_lvl = np.ma.median(np.ma.array(DAPI,mask=~mask)) - DAPI_bg
    
    CY5_bg = np.ma.median(np.ma.array(CY5,mask=~mask_bg))
    CY5_lvl = np.ma.median(np.ma.array(CY5,mask=~mask)) - CY5_bg

    FITC_bg = np.ma.median(np.ma.array(FITC,mask=~mask_bg))
    FITC_lvl = np.ma.median(np.ma.array(FITC,mask=~mask)) - FITC_bg
    
    return [DAPI_lvl,FITC_lvl,CY5_lvl]




def read_chip(xp,chip):
    

    Chip = pd.DataFrame(columns=columns2)
    path = '\\'.join((data_path,xp,chip,'Barcoding'))
    # path = '\\'.join((r'P:\Romain\data',xp,chip,'Barcoding'))
    for im_name in tqdm(os.listdir(path+'\\DAPI')):
        if im_name.endswith('tif'):
            well = im_name.split('_')[2][2:]
            Chip = Chip.append(pd.DataFrame([[chip,well,*read_well(path,im_name)]],\
                                            columns=columns2))
    
    
    
    Final_Chip = pd.DataFrame(columns=columns)
    Final_Chip.chip = Chip.chip
    Final_Chip.well = Chip.well
    
    for dye in ['CY5','DAPI']:
        try:
            Final_Chip[Dyes_Drugs[dye]] = dye_to_drug(Chip[dye.lower()], dye, Exposition[dye])
        except KeyError:
            pass
    Final_Chip.control = dye_to_control(Chip) 
    
    
    return Final_Chip



