from cmath import nan
from matplotlib.collections import PathCollection
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

def readExcel(path):
      """ func for reading Romain's xlsx data """
      try:
            df = pd.read_excel(path,
                  engine="openpyxl",
                        index_col=0,
                              dtype={'date': str} # we just specify the type of coumn for date
            
            ) # first columns are the indexed column
   
      except Exception as ve:

            print("cannot create df due to\n")
            print(f"{ve}")

            df = pd.DataFrame()
            
      return df



######################### analysis Drugs  #############################
def readDataChips():
      """ reading the data from E9 for Cisplatine, and E11 for Etoposide """
      
      dfCis = readPickel(path.join("E9","ChipsOK.pkl"))
      dfCisBarcode = readPickel(path.join("E9","Barcode.pkl"))

      dfEtoPickel = readPickel(path.join("E11","Chips.pkl"))
      dfEtoExcel = readExcel(path.join("E11","Chips.xlsx"))
      dfEtoBarcode = readPickel(path.join("E11","Barcode.pkl"))

      return dfEtoPickel,dfEtoBarcode,dfCis,dfCisBarcode

def readDataChipsExcel():
      """ reading the data from E9 for Cisplatine, and E11 for Etoposide """
      
      dfCis = readExcel(path.join("E9","Chips.xlsx"))
      dfCisBarcode = readExcel(path.join("E9","Barcode.xlsx"))

      dfEto = readExcel(path.join("E11","Chips.xlsx"))
      dfEtoBarcode = readPickel(path.join("E11","Barcode.xlsx"))

      return dfEto,dfEtoBarcode,dfCis,dfCisBarcode

def read_csv():
      """ function for opening the already treated data """

      pathCis = path.join("analysisGM","CisplatineE9_EtoposideE11","20220908_GManalysis_CisE9.txt")
      dfCis = pd.read_csv(pathCis, header=0,sep="\t")

      pathEto = path.join("analysisGM","CisplatineE9_EtoposideE11","20220908_GManalysis_EtoE11.txt")
      dfEto = pd.read_csv(pathEto, header=0,sep="\t")

      return dfEto, dfCis

def dropDuplicates(df1,df2):
      """function for removing duplicates for two dfs"""
      df1D = df1.drop_duplicates(subset=["date","chip","well","viability"])
      df2D = df2.drop_duplicates(subset=["date","chip","well","viability"])
      return df1D, df2D

def addDataDrugsChips(dfChip,dfBarcode, drugName):
      """ add a column corresponding to the concentration of drug """

      df = pd.merge(dfChip,dfBarcode,on=["chip","well"])

      df.rename({drugName: 'Cdrug'}, axis=1, inplace=True)
      df["drugName"] = drugName

      return df

def removeBadPoints(df,ctlName):
      """ function for removing bad values """

      # removin absurde detection
      df2 = df.copy()
      df2 = df2[(df2["radius"]>25)&(df2["radius"]<150)]

      # removing barcoding false detection
      points = df2[(df2["control"]!=ctlName)&(df2["Cdrug"]>5e-2)]
      ctl = df2[(df2["control"]==ctlName)]
      #ctl =  ctl[(ctl["Cdrug"]<1e-1) | (ctl["Cdrug"]==nan)] 
      df3 = pd.concat([points,ctl],join="inner",ignore_index=True)

      return df3, ctl

def addR0(df):
      """ create a column with the initial size of the spheroid 
            return dfR0 then df with the column      
      """

      dates = df.date.unique()
      print(f"dates:{dates}")
      dates.sort()
      D0 = dates[0] #getting D0

      # we get only the D0
      dfR0 = df[df["date"]==D0]
      # we keep only three columns
      dfR0 = dfR0[["chip","well","radius"]]
      dfR0.rename(columns={"chip":"chip","well":"well","radius":"R0"},inplace=True)

      # we merge
      df2 = df = pd.merge(df,dfR0,on=["chip","well"])

      return dfR0,df2

def addPooledConcentration(df):
      """ function add the centrale value of the concentrations """

      drugName = df.drugName.unique()[0]

      if drugName == "Cisplatin":
            # boundaries are 0.005µM and 50 µM
            suposedDrugsValue = [50,25,10,5,1,0.5,0.1,0.05,0.015,0.01,0.005]
            # not the same range
      else:
            # boundaries are 0.005µM and 100 µM
            suposedDrugsValue = [50,25,10,5,1,0.5,0.1,0.05,0.015,0.01,0.005]


      def filterConcentration1(c):
            """ filter the concentration and return a centrale value """

            filteredC = False

            for centraleC in suposedDrugsValue:
                  if c <= 1.33*centraleC and c >=0.66*centraleC:
                        print(f"C:{c}, centraleC:{centraleC}")
                        filteredC = centraleC

            if not filteredC:
                  print(f"\n No matching concentration for {c}")
            
            return filteredC
      
      def filterConcentration2(c):
            """ filter the concentration and return a centrale value """

            filteredC = False

            for i,centraleC in enumerate(suposedDrugsValue):
                  #increasing order
                  if i==0 and c>=centraleC:
                        print(f"C:{c}, centraleC:{centraleC}")
                        filteredC = centraleC
                  
                  elif centraleC == suposedDrugsValue[-1] and c<= centraleC:
                        print(f"C:{c}, centraleC:{centraleC}")
                        filteredC = centraleC

                  elif i!=0 and centraleC != suposedDrugsValue[-1]:
                        if c >= centraleC and c <= suposedDrugsValue[i-1]:
                              print(f"C:{c}, centraleC:{centraleC}")
                              filteredC = centraleC

            if not filteredC:
                  print(f"\n No matching concentration for {c}")
            
            return filteredC

      df["centraleC"] = df["Cdrug"].apply(filterConcentration2)


      return df

######## plots #####################

def plot_Chip(df,save=True,xlab = "Drug concentration (µM)",fileName ="",printFit = True):
      """Plots the viab in function of drug concentration (xData) and finds
      sigmoid fit"""

      drugName = df.drugName.unique()[0]

      dates = df.date.unique()
      dates.sort()
      fig,axes = plt.subplots(2,3,figsize=(20,11.50))
      D0 = datetime.strptime(dates[0],'%y%m%d')
      
      
      
      for a,date in zip(axes.ravel(),dates):

            # we filter according to date et non control
            dfData = df[(df["control"]==False)&(df["date"]==date)]
            dfCtl =  df[(df["control"]!=False)&(df["date"]==date)]

                  #Scatter viab with radius (color of the dots) and add controls in boxplot
            sns.scatterplot(data = dfData,x="Cdrug",y='viability',hue='R0',
                                    palette='viridis',alpha=.7,
                                          s = 50,
                                          linewidth=0,
                                          ax=a) #data in scatter
            a.boxplot(dfCtl.viability,positions=[50],widths=20) 

            if date!=dates[0]:
                        
                  if printFit:
                        lab = ""# we set a value if we want to see the result of the fit 
                  else:
                        lab = '_nolegend_'
                  fit_sigm(dfData.viability,dfData["Cdrug"], a,'red',1,lab=lab)   #fitting only for Di>0
                  
            a.set_xlabel(xlab,fontsize=18)
            a.set_ylabel('Viability',fontsize=18)
            a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}",fontsize=24)
            a.set_xscale('log')
            a.set_ylim((-.1,1.1))
            a.set_xlim((4e-2,2e2))
            a.tick_params(axis='both', which='major',labelsize=14,length=10)
            a.tick_params(axis='both', which='minor',labelsize=14,length=7)

            if printFit:
                  title=False
            else:
                  title = "Initial radius (µm)"
            a.legend(title=title, fontsize=14, markerscale=1.5)

      plt.tight_layout()
      if save:
            fileName = f"{fileName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_ChipD2(df,save=True,xlab = "Drug concentration (µM)",fileName ="",printFit = True):
      """Plots the viab in function of drug concentration (xData) and finds
      sigmoid fit"""
    
      drugName = df.drugName.unique()[0]

      dates = df.date.unique()
      dates.sort()
      fig,a = plt.subplots(figsize=(20,17))
      D0 = datetime.strptime(dates[0],'%y%m%d')
      
      date = dates[2]   

      # we filter according to date et non control
      dfData = df[(df["control"]==False)&(df["date"]==date)]
      dfCtl =  df[(df["control"]!=False)&(df["date"]==date)]
      
      #Scatter viab with radius (color of the dots) and add controls in boxplot
      sns.scatterplot(data = dfData,x="Cdrug",y='viability',hue='R0',
                        palette='viridis',
                              alpha=.7,
                                    s = 500,
                                    linewidth=0,#remove the line around the markers
                                          ax=a) #data in scatter
      a.boxplot(dfCtl.viability,positions=[50],widths=20) 


      if printFit:
            lab = ""# we set a value if we want to see the result of the fit 
      else:
            lab = '_nolegend_'
      fit_sigm(dfData.viability,dfData["Cdrug"], a,'red',1,lab=lab)   #fitting only for Di>0

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
            lgd = a.legend(title=title, loc=3, fontsize=56, markerscale=4)
            lgd.get_title().set_fontsize('50')# resize the legend title
            # #resizing the plots inside the legend 
            # for h in lgd.legendHandles:
            #       h._legmarker.set_markersize(400)
      plt.tight_layout()
      if save:
            fileName = f"{fileName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)


def plot_BoxplotCentraleC(df,drugName="Drug",save=True):
    """Plots the viab in function of drug concentration (xData) and finds
    sigmoid fit"""
    

    dates = df.date.unique()
    dates.sort()
    fig,axes = plt.subplots(2,3,sharex='all', figsize=(20,15))
    D0 = datetime.strptime(str(dates[0]),'%y%m%d')
    print("before loop")
    
      
    for a,date in zip(axes.ravel(),dates):
      
      # we filter according to date et non control
      dfData = df[(df["control"]=="False")&(df["date"]==date)&(df["centraleC"]!="False")]

      r0 = dfData["R0"]
      C = dfData["centraleC"]
      print(f"{len(r0)},{len(C)}")

      dfCtl =  df[(df["control"]==True)&(df["date"]==date)]
      
        #Scatter viab with radius (color of the dots) and add controls in boxplot
      #sns.scatterplot(data = dfData,x=f"{drugName}",y='viability',hue='R0',palette='viridis',alpha=.7,ax=a) #data in scatter
      print("before boxplot")
      sns.boxplot(data=dfData, x="centraleC", y="R0",hue="centraleC",palette='viridis',width=0.5,ax=a)
      
      a.set_xlabel(f'{drugName} concentration (µM)')
      a.set_ylabel('R0')
      a.set_title(f"D{(datetime.strptime(str(date),'%y%m%d')-D0).days}")
      a.set_xscale('log')
      a.set_ylim((0,100))
      a.set_xlim((4e-2,2e2))
      a.legend()

      if save:
            fileName = f"chip_plotBoxPlotCentraleC_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)


def plot_IC50_BoxPlotR0(df: pd.DataFrame,save=True):
    """Plots the viab in function of drug concentration (xData) and finds
    sigmoid fit"""
    
    drugName = df.drugName.unique()[0]
    dates = df.date.unique()
    dates.sort()
    print(dates)
    fig,axes = plt.subplots(2,3,sharex='all', figsize=(30,15))
    D0 = datetime.strptime(dates[0],'%y%m%d')
    
    for a,date in zip(axes.ravel(),dates):
        
      # we filter according to date et non control
      dfData = df[ (df["control"]==False)&(df["date"]==date) ]
      dfCtl =  df[ (df["control"]==True)&(df["date"]==date) ]

      #Scatter viab with radius (color of the dots) and add controls in boxplot
      sns.scatterplot(data = dfData,x="Cdrug",y='viability',hue='R0',palette='viridis',alpha=.7,ax=a) #data in scatter
      #ctl = a.boxplot(dfCtl.viability,positions=[50],widths=20) 
      ### we need to compute manually the width of the boxplots or they will be ugly
      width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
      w = 0.1

      a2 = a.twinx()
      
      positions = dfData["centraleC"].sort_values().unique()
      dataBoxPlot = []
      for pos in positions:
            dataBoxPlot.append(dfData[df["centraleC"] == pos].R0.values.tolist())
            # so we create sublists with the data for plotting
      a2.boxplot(dataBoxPlot,positions=positions,  widths= width(positions,w)) 
      #maybe plt.boxplot is not idle, but at least it has a widths parameter
      if date!=dates[0]:
            fit = fit_sigm(dfData.viability,dfData["Cdrug"], a,'red',1,1)   #fitting only for Di>0

      
      a.set_xlabel(f'{drugName} concentration (µM)')
      a.set_ylabel('Viability')
      a.set_title(f"D{(datetime.strptime(str(date),'%y%m%d')-D0).days}")
      a.set_xscale('log')
      a.set_ylim((-.1,1.1))
      a.set_xlim((4e-2,2e2))

      a2.set_ylabel("Initial radius (µm)")
      a2.set_ylim((20,80))

      a.legend(loc=3)
      a2.legend(loc=0)
      

      if save:
            fileName = f"chip_plotIC50_BarplotR0_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_Chips(dfEto,dfCis,save=True):
    """Plots the graph for both drugs"""
    
    plot_Chip(dfEto,"Etoposide",save=save)
    plot_Chip(dfCis,"Cisplatin",save=save)

def plot_Chip_R0_vs_Viab(df,drugName="Drug",save=True):
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

            sns.scatterplot(data = dfData,x="R0",y='viability',hue=f"{drugName}",palette='viridis',alpha=.7,ax=a) #data in scatter
            a.set_xlabel("Initial radius (µm)")
            a.set_ylabel('Viability')
            a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
            #a.set_xscale('log')
            # a.set_ylim((-.1,1.1))
            # a.set_xlim((4e-2,2e2))
            a.legend(title=f"{drugName} concentration (µM)")

      if save:
            fileName = f"Chip_R0_VS_Viab_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_Chip_R0_vs_Cdrug(df,drugName="Drug",save=True):
      """

      """
    
      dates = df.date.unique()
      dates.sort()
      fig,axes = plt.subplots(2,3,sharex='all', figsize=(20,15))
      D0 = datetime.strptime(dates[0],'%y%m%d')

      for a,date in zip(axes.ravel(),dates):
            dfData = df[(df["control"]==False)&(df["date"]==date)]
            dfCtl =  df[(df["control"]==True)&(df["date"]==date)]

            sns.scatterplot(data = dfData,x=f"{drugName}",y='R0',
                              hue=dfData["viability"],palette='viridis',alpha=.7,ax=a) #data in scatter
            a.set_xlabel("Cdrug (µM)")
            a.set_ylabel("Initial radius (µm)")
            a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
            a.set_xscale('log')
            # a.set_ylim((-.1,1.1))
            a.set_xlim((4e-2,2e2))
            a.legend(title="Viability")
      
      if save:
            fileName = f"Chip_R0_VS_Cdrug_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def plot_Chip_R0_vs_Cdrug_CcentralBoxplot(df,drugName="Drug",save=True):
      """

      """
    
      dates = df.date.unique()
      dates.sort()
      fig,axes = plt.subplots(2,3,sharex='all', figsize=(20,15))
      D0 = datetime.strptime(str(dates[0]),'%y%m%d')

      for a,date in zip(axes.ravel(),dates):
            dfData = df[(df["control"]=="False")&(df["date"]==date)]
            dfR0 = df[(df["control"]=="False")&(df["date"]==date)&(df["centraleC"]!=False)&(df["R0"]!=False)]
            dfCtl =  df[(df["control"]=="True")&(df["date"]==date)]

            sns.scatterplot(data = dfData,x=f"{drugName}",y='R0',
                              hue=dfData["viability"],palette='viridis',alpha=.7,ax=a) #data in scatter
            
            dfR0.boxplot(column=["R0"],by="centraleC",positions=dfR0.centraleC.unique(),ax=a) 

            a.set_xlabel("Cdrug (µM)")
            a.set_ylabel("Initial radius (µm)")
            a.set_title(f"D{(datetime.strptime(str(date),'%y%m%d')-D0).days}")
            a.set_xscale('log')
            # a.set_ylim((-.1,1.1))
            a.set_xlim((4e-2,2e2))
            a.legend(title="Viability")
      
      if save:
            fileName = f"Chip_R0_VS_Cdrug_BoxPlotCcentral_{drugName}.svg"
            p = path.join("analysisGM",fileName)
            plt.savefig(p)

def dataToCSV(df,name="data"):

      """ function for saving the data as txt """
      fileName = f"{name}.txt"
      p = path.join("analysisGM",fileName)
      df.to_csv(p,header=True,index=None,sep="\t",mode="a")


def mainChip():

      """ function to run for obtaining the IC50 for the plates """
      #reading the data
      dfEto, dfEtoBarcode, dfCis, dfCisBarcode = readDataChips()
      print("\nCombining with barcode data")
      dfEtoFull = addDataDrugsChips(dfEto,dfEtoBarcode,"Etoposide")
      dfCisFull = addDataDrugsChips(dfCis,dfCisBarcode,"Cisplatin")
      print("\nRemoving bad values")
      print("\For eto")
      dfEtoFull2 = removeBadPoints(dfEtoFull)
      print("\nNow for cis")
      dfCisFull2 = removeBadPoints(dfCisFull)      

      #adding R0: 
      print("\nAdding R0 for Eto")
      dfEtoR0,dfEtoFull = addR0(dfEtoFull)
      print("\nAdding R0 for Cis")
      dfCisR0,dfCisFull = addR0(dfCisFull)
      # dropping the duplicates
      dfEtoFull, dfCisFull = dropDuplicates(dfEtoFull,dfCisFull)
      # adding the centrale values of concentrations 
      dfCisFull2 = addPooledConcentration(dfCisFull,"Cisplatin")
      dfEtoFull2 = addPooledConcentration(dfEtoFull,"Etoposide")

      #plotting
      # plot_IC50_BoxPlotR0(dfCis2,drugName="Cisplatin",save=True)
      # plot_IC50_BoxPlotR0(dfEto2,drugName="Etoposide",save=True)
      



