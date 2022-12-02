from cmath import nan
import pandas as pd
from os import path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from Plot import fit_sigm

def readPickel(path):
      """ function for read Romain's pickeled data """
      try:
            df = pd.read_pickle(path)
      except Exception as ve:

            print("cannot create df due to\n")
            print(f"{ve}")

            df = pd.DataFrame()
            
      return df

######################### analysis Plates #############################
def readDataPlate():
      """ reading the data from E4 """
      dfPlatesOK = readPickel(path.join("E4","PlatesOK.pkl"))
      print(dfPlatesOK.head(10))

      dfPlates = readPickel(path.join("E4","Plates.pkl"))
      print(dfPlates.head(10))

      dfD0_ok = readPickel(path.join("E4","D0_ok.pkl"))
      print(dfD0_ok.head(10))

      dfD0_ok = readPickel(path.join("E4","D0.pkl"))
      print(dfD0_ok.head(10))

      dfThresholds = readPickel(path.join("E4","Thresholds.pkl"))

def addDataDrugsPlates(df):
      """ add a column corresponding to the concentration of drug """

      # concentrations used both for cisPlatine and etoposide
      Cdrug = [100,46.4,21.5,10,4.64,2.15,1,0.464,0.215,0.1] #µM

      colDrug = []
      row = df.row
      column = df.column

      # we create a column corresponding to the drug 
      for r,c in zip(row,column):

            if str(r) != "F":
                  # F is the controle 
                  colDrug.append(Cdrug[int(c)-1])
            else:
                  colDrug.append("ctl")
      
      # we create a new column 
      df["Cdrug"] = colDrug

      # we create a new column with the name of the drug
      #only two plates: E4MW01: CisPlat
      # E4MW02 : Etoposide
      exp = df.plate_name
      colDrug = []

      for _ in exp:
            if _ == "E4MW01":
                  colDrug.append("CisPlat")
            else:
                  colDrug.append("Eto")
      
      # we create the corresponding column 

      df["drug_name"] = colDrug

      return df

def splitPlate(df):

      dfEto = df[ df["drug_name"]=="Eto" ]
      dfCisPlat = df[ df["drug_name"]=="CisPlat" ]

      return dfEto,dfCisPlat

def addR0(df):
      """ create a column with the initial size of the spheroid """

      dates = df.date.unique()
      print(f"dates:{dates}")
      dates.sort()
      D0 = dates[0] #getting D0
      D1 = dates[1]

      colR0Cis = [] # column with the radius of D0
      colR0eto = []
      colDate = df.date
      colRad = df.Radius
      colDrug = df.drug_name
      

      for d,r,drug in zip(colDate,colRad,colDrug):
            if drug == "CisPlat":
                  if d == D0:
                        print("append Cis")
                        if r < 300:
                              colR0Cis.append(r)
                        else:
                              colR0Cis.append("nan")
                  # take the value for D1 to remove a detection issue
                  
            
            elif drug == "Eto":
                  if d == D0:
                        print("append Eto")
                        if r < 300:
                              colR0eto.append(r)   
                        else:
                              colR0eto.append("nan")

            else:
                  print("issue with the name of the drug")              

      # we have th column with all the Initial radius we extand it 

      colFinalR0cis = []
      colFinalR0eto = []

      # not elangant, we concatenated according to the number of dates
      for i in range(len(dates)):
            colFinalR0cis += colR0Cis
            colFinalR0eto += colR0eto
      
      # we create the df for eto and cis 
      dfEto,dfCisPlat = splitPlate(df)
      #we create the columns

      dfEto["R0"] = colFinalR0eto
      dfCisPlat["R0"] = colFinalR0cis

      return dfEto, dfCisPlat

def addR02(df):
      """ create a column with the initial size of the spheroid """

      dates = df.date.unique()
      print(f"dates:{dates}")
      dates.sort()
      D0 = dates[0] #getting D0

            # we get only the D0
      dfR0 = df[df["date"]==D0]
      # we keep only three columns
      dfR0 = dfR0[["plate_name","row","column","Radius"]]

      dfR0.rename(columns={"plate_name":"plate_name","row":"row","column":"column","Radius":"R0"},inplace=True)
      # we merge
      df2 = df = pd.merge(df,dfR0,on=["plate_name","row","column"])

      return df2
     

def plot_Plate(df,drugName="Drug",xlab = "Drug concentration (µM)",save=True,fileName ="",printFit = True):
    """Plots the viab in function of drug concentration (xData) and finds
    sigmoid fit"""
    
    dates = df.date.unique()
    dates.sort()
    fig,axes = plt.subplots(2,3,sharex=False,sharey=False, figsize=(20,15))
    
    D0 = datetime.strptime(dates[0],'%y%m%d')
    
    
    for a,date in zip(axes.ravel(),dates):
      
      # we filter according to date et non control
      dfData = df[(df["control"]==False)&(df["date"]==date)]
      dfCtl =  df[(df["control"]==True)&(df["date"]==date)]
      
        #Scatter viab with radius (color of the dots) and add controls in boxplot
      sns.scatterplot(data = dfData,x="Cdrug",y='Viability',hue='R0',palette='viridis',alpha=.7,ax=a,legend='auto') #data in scatter
      a.boxplot(dfCtl.Viability,positions=[50],widths=20) 
      
      if date!=dates[0]:
            if printFit:
              lab = ""# we set a value if we want to see the result of the fit 
            else:
              lab = '_nolegend_'
            fit_sigm(dfData.Viability,dfData.Cdrug, a,'red',1,lab=lab)   #fitting only for Di>0

      
      a.set_xlabel(xlab)
      a.set_ylabel('Viability')
      a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
      a.set_xscale('log')
      a.set_ylim((-.1,1.1))
      a.set_xlim((4e-2,2e2))
      
      if printFit:
        title=False
      else:
        title = "Initial radius (µm)"
      a.legend(title=title)

    plt.tight_layout()

    if save:
          fileName = fileName + ".svg"
          p = path.join("analysisGM",fileName)
          plt.savefig(p)

def plot_PlateD2(df,drugName="Drug",xlab = "Drug concentration (µM)",save=True,fileName ="",printFit = True):
      """Plots the viab in function of drug concentration (xData) and finds
      sigmoid fit"""

      dates = df.date.unique()
      dates.sort()
      fig,a = plt.subplots(figsize=(20,17))

      D0 = datetime.strptime(dates[0],'%y%m%d')
      date = dates[2]


      # we filter according to date et non control
      dfData = df[(df["control"]==False)&(df["date"]==date)]
      dfCtl =  df[(df["control"]==True)&(df["date"]==date)]

            #Scatter viab with radius (color of the dots) and add controls in boxplot
      sns.scatterplot(data = dfData,x="Cdrug",y='Viability',
                        hue='R0',
                              palette='viridis',
                                    alpha=.7,ax=a,
                                    s = 500, linewidth=0) #data in scatter
      a.boxplot(dfCtl.Viability,positions=[50],widths=20) 

      
      if printFit:
            lab = ""# we set a value if we want to see the result of the fit 
      else:
            lab = '_nolegend_'
      fit_sigm(dfData.Viability,dfData.Cdrug, a,'red',1,lab=lab)   #fitting only for Di>0


      a.set_xlabel(xlab,fontsize=78)
      a.set_ylabel('Viability',fontsize=78)
      a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}",fontsize=78)
      a.set_xscale('log')
      a.set_ylim((-.1,1.1))
      a.set_xlim((4e-2,2e2))
      a.tick_params(axis='both', which='major',labelsize=45,length=20)
      a.tick_params(axis='both', which='minor',labelsize=45,length=10)

      if printFit:
            title=False
      else:
            title = "Initial radius (µm)"
            lgd = a.legend(title=title)
            lgd = a.legend(title=title,loc=3,fontsize=56,markerscale=4)
            lgd.get_title().set_fontsize('50')

      plt.tight_layout()

      if save:
            fileName = f"{fileName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_Plates(dfEto,dfCis,save=True):
    """Plots the graph for both drugs"""
    
    plot_Plate(dfEto,"Etoposide",save=save)
    plot_Plate(dfCis,"Cisplatin",save=save)

def plot_R0_vs_Viab(df,drugName="Drug",save=True):
      """
      attempt to plot Viab against R0 along the time

      """
    
      dates = df.date.unique()
      dates.sort()
      fig,axes = plt.subplots(2,3,sharex='all', figsize=(20,15))
      D0 = datetime.strptime(dates[0],'%y%m%d')

      for a,date in zip(axes.ravel(),dates):
            dfData = df[(df["control"]==False)&(df["date"]==date)]
            dfCtl =  df[(df["control"]==True)&(df["date"]==date)]

            sns.scatterplot(data = dfData,x="R0",y='Viability',hue="Cdrug",palette='viridis',alpha=.7,ax=a) #data in scatter
            a.set_xlabel("Initial radius (µm)")
            a.set_ylabel('Viability')
            a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
            #a.set_xscale('log')
            # a.set_ylim((-.1,1.1))
            # a.set_xlim((4e-2,2e2))
            a.legend(title=f"{drugName} concentration (µM)")

      if save:
            fileName = f"Plate_R0_VS_Viab_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_R0_vs_Cdrug(df,drugName="Drug",save=True):
      """

      """
    
      dates = df.date.unique()
      dates.sort()
      fig,axes = plt.subplots(2,3,sharex='all', figsize=(20,15))
      D0 = datetime.strptime(dates[0],'%y%m%d')

      for a,date in zip(axes.ravel(),dates):
            dfData = df[(df["control"]==False)&(df["date"]==date)]
            dfCtl =  df[(df["control"]==True)&(df["date"]==date)]

            sns.scatterplot(data = dfData,x="Cdrug",y='R0',
                              hue=dfData["Viability"],palette='viridis',alpha=.7,ax=a) #data in scatter
            a.set_xlabel("Cdrug (µM)")
            a.set_ylabel("Initial radius (µm)")
            a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
            a.set_xscale('log')
            # a.set_ylim((-.1,1.1))
            a.set_xlim((4e-2,2e2))
            a.legend(title="Viability")
      
      if save:
            fileName = f"Plate_R0_VS_Cdrug_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def dataToCSV(df,name="data"):

      """ function for saving the data as txt """
      fileName = f"{name}.txt"
      p = path.join("analysisGM",fileName)
      df.to_csv(p,header=True,index=None,sep="\t",mode="a")


def mainPlate():

      """ function to run for obtaining the IC50 for the plates """
      #reading the data
      df = readPickel(path.join("E4","PlatesOK.pkl"))
      #adding the column corresponding to the drugs
      df = addDataDrugsPlates(df)
      # spliting btw etoposide and cisplatine, plus adding column of R0
      dfEto,dfCisPlat = addR0(df)
      #we save the data
      # dataToCSV(dfEto,name="20220908_E4_PlatesOK_Etoposide_IC50_R0")
      # dataToCSV(dfCisPlat,name="20220908_E4_PlatesOK_CisPlat_IC50_R0")
      # #we plot and save the plot 
      # plot_Plates(dfEto,dfCisPlat)

      return df,dfEto,dfCisPlat

