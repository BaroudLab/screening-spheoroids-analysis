# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:37:59 2021

@author: Romain
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from skimage.measure import regionprops,label
from skimage.morphology import opening,closing,disk

#%%

columns = ['chip','date','well','radius','gfp','pi','viability']


#%% Well functions


# Read values of fluo of a well 

def read_well(path,well,thresh=None,norm=None):
    """
    Parameters
    ----------
    path : path to BF/FITC/TRIC diretories (usually date diretory).\n
    im_name : name of FITC image.

    Returns
    -------
    Rad : effectiv radius of the spheroid (um).\n
    GFP_lvl : summed GFP lvl (bg-corrected) over spheroid.\n
    PI_lvl : same as GFP for PI.\n
    viab : viability of the spheroid.
    """
    
    if not thresh: #value not passed --> findthresh is calling, reading only PI
        
        PI = cv2.imread(os.path.join(path,"".join([well,"CY3",".tif"]) ),-1 )
        h,w = np.shape(PI)
        center = (w//2,h//2)      
        Y, X = np.ogrid[:h, :w]
        
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    
        maskcenter = dist_from_center <= .7*h        
        
        PII = np.ma.array(PI,mask=~maskcenter)
        thresh = np.ma.median(PII)+2*np.ma.std(PII)
        norm = np.ma.mean(np.ma.array(PI,mask=~(PI>thresh)))+\
            2*np.std(np.ma.array(PI,mask=~(PI>thresh)))
        
        #return thresh,norm
    
    if not norm: norm=thresh #value passed --> read_chip is calling, reading all fluo, find viab
    
    #reading pictures
    GFP = cv2.imread( os.path.join(path,"".join([well,"GFP",".tif"]) ) ,-1)

    PI = cv2.imread (os.path.join(path,"".join([well,"CY3",".tif"]) ),-1 )
    
    #making circ. mask around center
    h,w = np.shape(GFP)
    center = (w//2,h//2)      
    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    maskcenter = dist_from_center <= h        
    
    #masks based on fluo, above med+3std
    maskGFP = (GFP>(np.median(GFP)+3*np.std(GFP)))&maskcenter   
    maskPI = (PI>thresh)&maskcenter 
    mask = (maskGFP|maskPI)&maskcenter
    
    #Some cleaning
    mask = opening(mask,disk(2))
    mask = closing(mask,disk(2))
    mask = label(mask)
    
    #finding different spheroids (not perfect, maybe should use only gfp channel? 
    #but loosing full dead spheros...)
    spheroids = regionprops(mask)
    
    Res = pd.DataFrame(columns=['well','radius','gfp','pi','viability'])
    
    # Finding viability and radius for all spheroids in well
    for sphero in spheroids:
        
        if 5<sphero.equivalent_diameter*.65/2<300: #avoiding too-little or full-well spheros
            
            L = sphero.label
    
            Rad = np.sqrt(np.sum(maskGFP*(mask==L))/np.pi)*.65
            GFP_bg = np.ma.mean(np.ma.array(GFP,mask=mask))
            GFP_lvl = np.ma.sum(np.ma.array(GFP - GFP_bg,mask=~(mask==L))) 
            
            
             
            PI_bg = np.ma.mean(np.ma.array(PI,mask=mask))
            PI_lvl = np.ma.sum(np.ma.array(PI - PI_bg,mask=~(mask==L))) 
      
            if np.ma.sum(maskPI*(mask==L))==0:viab = 1
            
            else:viab = 1-np.ma.sum(np.ma.array(PI-PI_bg,mask=~(maskPI*(mask==L)))**.3)/(norm**.3*np.sum(mask==L))
     
        
            
            Res.loc[-1] = [well,Rad,GFP_lvl,PI_lvl,viab]
    print(Res.head())
    
    return Res

def read_well_from_okovision(well,trapGFP,trapCY3,thresh=None,norm=None):
    """
    Parameters
    ----------
    path : path to BF/FITC/TRIC diretories (usually date diretory).\n
    im_name : name of FITC image.

    Returns
    -------
    Rad : effectiv radius of the spheroid (um).\n
    GFP_lvl : summed GFP lvl (bg-corrected) over spheroid.\n
    PI_lvl : same as GFP for PI.\n
    viab : viability of the spheroid.

    patched to be use with chip objects from Okovision
    """
    
    GFP = trapGFP
    PI = trapCY3

    if not thresh: #value not passed --> findthresh is calling, reading only PI
        
        h,w = np.shape(PI)
        center = (w//2,h//2)      
        Y, X = np.ogrid[:h, :w]
        
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    
        maskcenter = dist_from_center <= .7*h        
        
        PII = np.ma.array(PI,mask=~maskcenter)
        thresh = np.ma.median(PII)+2*np.ma.std(PII)
        norm = np.ma.mean(np.ma.array(PI,mask=~(PI>thresh)))+\
            2*np.std(np.ma.array(PI,mask=~(PI>thresh)))
        
        #return thresh,norm
    
    if not norm: norm=thresh #value passed --> read_chip is calling, reading all fluo, find viab
    

    #making circ. mask around center
    h,w = np.shape(GFP)
    center = (w//2,h//2)      
    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    maskcenter = dist_from_center <= h        
    
    #masks based on fluo, above med+3std
    maskGFP = (GFP>(np.median(GFP)+3*np.std(GFP)))&maskcenter   
    maskPI = (PI>thresh)&maskcenter 
    mask = (maskGFP|maskPI)&maskcenter
    
    #Some cleaning
    mask = opening(mask,disk(2))
    mask = closing(mask,disk(2))
    mask = label(mask)
    
    #finding different spheroids (not perfect, maybe should use only gfp channel? 
    #but loosing full dead spheros...)
    spheroids = regionprops(mask)

    Res = pd.DataFrame(columns=['radius','gfp','pi','viability','thresh','norm'])
    
    # Finding viability and radius for all spheroids in well
    i = 0
    for sphero in spheroids:
        
        if 5<sphero.equivalent_diameter*.65/2<300: #avoiding too-little or full-well spheros
            
            L = sphero.label
    
            Rad = np.sqrt(np.sum(maskGFP*(mask==L))/np.pi)*.65
            GFP_bg = np.ma.mean(np.ma.array(GFP,mask=mask))
            GFP_lvl = np.ma.sum(np.ma.array(GFP - GFP_bg,mask=~(mask==L))) 
            
            
             
            PI_bg = np.ma.mean(np.ma.array(PI,mask=mask))
            PI_lvl = np.ma.sum(np.ma.array(PI - PI_bg,mask=~(mask==L))) 
      
            if np.ma.sum(maskPI*(mask==L))==0:viab = 1
            
            else:viab = 1-np.ma.sum(np.ma.array(PI-PI_bg,mask=~(maskPI*(mask==L)))**.3)/(norm**.3*np.sum(mask==L))


            new_row = {'well':well,'radius':Rad,'gfp':GFP_lvl,'pi':PI_lvl,'viability':viab,
                            'thresh':thresh, 'norm':norm
                                            }
            dfNewRow = pd.DataFrame(data = new_row, index = [i])

            Res = pd.concat([Res,dfNewRow],ignore_index=True)

            i += 1

    
    print(f"\n Done for well {well}")
    return Res,maskGFP,maskPI,mask

#Findind threshold and norm for PI
def findThresh(save_path,chips,date,xp):    

    
    for chip in chips:        
        path = '\\'.join((r'\\gaia.pasteur.fr\Multicell\Romain\data',xp,chip,date))
        
        
        PI_thresh = []
        PI_norm = []            
        for im_name in tqdm(os.listdir(path+r'\TRITC')):
            if im_name.endswith('tif'):
                t,n = read_well(path,im_name)
                PI_thresh.append(t)
                PI_norm.append(n)
                    
        PI_thresh = np.nanquantile(PI_thresh,.75)
        PI_norm = np.nanquantile(PI_norm,.75)
        
        try:
            Threshs = pd.read_pickle(save_path+"\\Thresholds.pkl")
                        
        except FileNotFoundError:
            Threshs = pd.DataFrame(columns=["chip","thresh",'norm','date'])
            
                
        Threshs = Threshs.append(pd.DataFrame([[chip,PI_thresh,PI_norm,date]],columns=Threshs.columns))
        
        pd.to_pickle(Threshs, save_path+"\\Thresholds.pkl")
        
#%% Reading all the chip


def read_chip(xp,chip,date,save_path):
    
    
    Chip = pd.DataFrame(columns=columns)
    path = '\\'.join((r'\\gaia.pasteur.fr\Multicell\Romain\data',xp,chip,date))
    
    Threshs = pd.read_pickle(save_path+"\\Thresholds.pkl")
    PI_thresh = np.max(Threshs[Threshs.chip==chip].thresh)
    PI_norm = np.max(Threshs[Threshs.chip==chip].norm)
    
    PI_thresh = 3.5e3  #Typical values that work fine in case of bad thresholds found
    PI_norm = 14e3
        

    
    for im_name in tqdm(os.listdir(path+'\\FITC')):
        if im_name.endswith('tif'):
            well = im_name.split('_')[2][2:]
            Temp = read_well(path,im_name,PI_thresh,PI_norm )
            Temp['chip']=chip
            Temp['well']=well
            Temp['date']=date
            Chip = Chip.append(Temp)
            
                
    return Chip