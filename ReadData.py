# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:57:07 2021

@author: stage
"""

from pandas import DataFrame, read_pickle
from tqdm import tqdm
import numpy as np
import os 

#os.chdir(r'\\gaia.pasteur.fr\Multicell\Romain\Programmes\ChipsAnalysis')
from GetDataBis import findThresh
from GetDataBis import read_chip as Data_Chip
from GetDrugs import read_chip as Drug_Chip
from GetDrugs import columns as bar_columns


    

#%% 

def read_threshs(xp,dates,chips,save_path,columns):
    
    tqdm.write("\n__Reading thresholds__\n")
    
    try:
        Threshs = read_pickle(save_path+"\\Thresholds.pkl")
        for date in dates:
            if len(Threshs[(Threshs.date==date)])<len(chips):
                findThresh(save_path,chips,date,xp)
    except FileNotFoundError:
        for date in dates:
            findThresh(save_path,chips,date,xp)
        
    Threshs =read_pickle(save_path+"\\Thresholds.pkl") 


def read_bio(xp,dates,chips,save_path,columns):
    
     
    
    tqdm.write("\n__Reading viability__\n")

    
    try:
        Chips = read_pickle(save_path+"\\Chips.pkl")        
                    
    except (FileNotFoundError,KeyError):
        Chips = DataFrame(columns=columns)
        for date in dates:
            for chip in chips:
                Chips = Chips.append(Data_Chip(xp, chip, date,save_path))
                Chips.to_pickle(save_path+"\\Chips.pkl")    
                
    for date in dates:
            for chip in chips:
                if len(Chips[(Chips.date==date)&(Chips.chip==chip)])==0:
                    Chips = Chips.append(Data_Chip(xp, chip, date,save_path))
                    Chips.to_pickle(save_path+"\\Chips.pkl")    
            
        
                
    Chips.to_pickle(save_path+"\\Chips.pkl")    
    
    return Chips



def read_barcode(xp,dates,chips,save_path,columns):
    
    
    
    tqdm.write("\n__Reading barcoding data__\n")
    
    try:
        Barcode = read_pickle(save_path+"\\Barcode.pkl")
        for chip in chips:
            if len(Barcode[(Barcode.chip==chip)])==0:
                Barcode = Barcode.append(Drug_Chip(xp, chip))
                    
    except (FileNotFoundError,KeyError):
        Barcode = DataFrame(columns=bar_columns)
        for chip in chips:
                Barcode = Barcode.append(Drug_Chip(xp, chip))
        
    
    Barcode.to_pickle(save_path+"\\Barcode.pkl")    
    
    return Barcode