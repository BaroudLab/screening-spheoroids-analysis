# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:23:14 2021

@author: Romain 
"""


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit


plt.rcParams['font.size'] = '14'
plt.rcParams['svg.fonttype']='none'


#%% Sigmoid fit

def sigm(x,L ,x0, k, b):
    """sigmoid function"""
    return (L-b) / (1 + np.exp(k*(np.log10(x)-np.log10(x0)))) + b



def find_params(ydata,xdata):
    """Fitting a sigmoid to ydata/xdata"""
    
    xdata,ydata = np.array(xdata),np.array(ydata)
    
    #clearing the datas
    xdata = np.array([x  if not np.isnan(x)  else 1e-1  for i,x in enumerate(xdata) if not np.isnan(ydata[i])])
    
    ydata = np.array([y for y in ydata if not np.isnan(y)])
    
    #intitial conditions for fit
    p0 = [1, 1,.1,0]
    
    try:   #fitting the data with curve_fit     
        popt,pcov = curve_fit(sigm, xdata, ydata,p0,
                              bounds=([.5,0,0,0],[1,100,1e3,.4]))#, method='dogbox')
    except ValueError:
        print('sthing happened')
        popt,pcov = curve_fit(sigm, xdata, ydata,p0, method='dogbox')
        
    
   
    return popt



def fit_sigm(ydata,xdata,figure,col='red',ic=True,lab=''):
    """
    Fitting a sigmoid to ydata/xdata and plot it in figure\n
    col: color of the plot (default red)\n
    ic: plot the vertical bar at IC50 value (default true)
    lab: adding a label with value of IC50 in plot (default false)
    """
    
    # xdata,ydata = np.array(xdata),np.array(ydata)
    
    # #clearing the datas
    # xdata = np.array([x  if not np.isnan(x)  else 1e-1 \
    #                   for i,x in enumerate(xdata) if not np.isnan(ydata[i])])
    # ydata = np.array([y for y in ydata if not np.isnan(y)])
    
    # #intitial conditions for fit
    # p0 = [1, 1,.1,0]
    
    # try:   
    #     popt,pcov = curve_fit(sigm, xdata, ydata,p0,
    #                           bounds=([.5,0,0,0],[1,100,1e3,.4]))
    # except ValueError:
    #     print('something happened during fit')
    #     popt,pcov = curve_fit(sigm, xdata, ydata,p0, method='dogbox')
    
    popt = find_params(ydata, xdata)
        
    
    #plot the sigmoid curve and the IC50
    x = np.logspace(np.log10(min(xdata)),np.log10(max(xdata)),500)
    y = sigm(x,*popt)
    
    figure.plot(x,y,color=col, label=lab)
    
    x05 = popt[1]
    print(popt)
    
    # custom the plot
    if lab != "_nolegend_":
        lab=f'IC50={np.round(x05,2)}'
    if ic:  
        figure.plot([x05]*2,[popt[3],popt[0]],color=col,label=lab)

    return x05




#%%
#
#
#
#%% Single drug experiments


def plot_drug(Chips,Barcode,xData):
    """Plots the viab in function of drug concentration (xData) and finds
    sigmoid fit"""
    
    dates = Chips.date.unique()
    dates.sort()
    fig,axes = plt.subplots(2,3,sharex='all')
    D0 = datetime.strptime(dates[0],'%y%m%d')
    
    for a,date in zip(axes.ravel(),dates):
        
        #Merging Barcode data and viab data (Chips)
        
        FullData = pd.merge(Barcode,Chips[(Chips.date==date)&(Chips.radius<150)&(Chips.radius>25)],on=['well','chip'])
        Data0 = pd.merge(Barcode,Chips[(Chips.date==dates[0])&(Chips.radius<150)&(Chips.radius>25)],on=['chip','well'])

        FullData['D0'] = Data0.viability
        
        # Controlling the values of drug. Too little --> control (barcode missreading)
        mask = (FullData[xData]>5e-2)&(FullData.control==False)
        maskCtrl = (FullData.control!=False)&(FullData[xData]<1e-1)
        
        Data = FullData[mask].copy()
        DataCtrl = FullData[maskCtrl].copy()
  
        #Scatter viab with radius (color of the dots) and add controls in boxplot
        sns.scatterplot(data = Data,x=xData,y='viability',hue='radius',palette='viridis',alpha=.7,ax=a) #data in scatter
        a.boxplot(DataCtrl.viability,positions=[50],widths=20) 
        
        if date!=dates[0]:fit_sigm(Data.viability,Data[xData], a,'red',1,1)   #fitting only for Di>0
                
        a.set_xlabel(f'{xData} concentration (um)')
        a.set_ylabel('Viability')
        a.set_title(f"D{(datetime.strptime(date,'%y%m%d')-D0).days}")
        a.set_xscale('log')
        a.set_ylim((-.1,1.1))
        a.set_xlim((4e-2,2e2))
        a.legend()



#%% Combinatorial experiments
# 
# 
# 
#%% 


def GetAndFilter(Chips,Barcode,date = None):
    """ Controlling the values of drug. Too little --> control (barcode missreading)\n
    Controlling spheroids radii: too little, not ok, too big, not ok"""
    
    Data = pd.merge(Barcode,Chips[(Chips.radius<=200)&(Chips.radius>=25)],on=['well','chip'])
    
    
    Data.Cisplatin = [c if (not np.isnan(c) and c>.015) else 3e-3*np.random.normal(1,.1) for c in Data.Cisplatin]
    Data.Etoposide = [e if (not np.isnan(e) and e>3e-3) else .9e-3*np.random.normal(1,.1) for e in Data.Etoposide]
    if date: return Data[(Data.date==date)]
    else: return Data


def day(date,dates):
    """from date to string D0/D1/D2..."""
    return f"D{(datetime.strptime(date,'%y%m%d')-datetime.strptime(dates[0],'%y%m%d')).days}"

#%% Plotting barcode

def color(Data):
    """ Color scale, gray to blue (eto), gray to red (cis), and mix (cis+eto)"""
    E,e = np.log10(np.max(Data.Etoposide)),np.log10(np.min(Data.Etoposide))
    E-=e
    C,c = np.log10(np.max(Data.Cisplatin)),np.log10(np.min(Data.Cisplatin))
    C-=c
    
    res = np.dstack((((np.log10(Data.Cisplatin)-c)/C),[0]*len(Data),(np.log10(Data.Etoposide)-e)/E))
    
    return [tuple(r) for r in res[0]]
    

def plot_barcode(Barcode):
    """ displays barcode with colors in function of the two drugs concentrations.\n
    Add marginal plots to read single drug conditions"""

    Data = Barcode.copy()
    Data.control = [0 if not(np.isnan(c) or np.isnan(e)) else 1 for c,e in zip(Barcode.Cisplatin,Barcode.Etoposide)]
    Data.Cisplatin = [c if (not np.isnan(c) and c>.015) else 3e-3*np.random.normal(1,.2) for c in Barcode.Cisplatin]
    Data.Etoposide = [e if (not np.isnan(e) and e>3e-3) else .5e-3*np.random.normal(1,.2) for e in Barcode.Etoposide]
    
    
    bins = np.logspace(-3.5,1.5,80)
    g = sns.JointGrid(x='Etoposide',y='Cisplatin',data=Data,)
    g.plot_joint(plt.scatter, edgecolor="white",alpha=.3,c=color(Data),s=100)
    _ = g.ax_marg_x.hist(Data.Etoposide,bins=bins)
    bins = np.logspace(-3,1.5,80)
    _ = g.ax_marg_y.hist(Data.Cisplatin, orientation="horizontal",bins=bins)
    
    
    
    
    g.ax_marg_x.set_xscale('log')
    g.ax_marg_y.set_yscale('log')

#%% Heatmap of # of spheroids per condition 
    
def plot_replicates(Chips,Barcode,cut_eto,lab_eto,cut_cis,lab_cis):
    
    fig,axes = plt.subplots(2,3, figsize=(50,100), squeeze= False)
    dates = Chips.date.unique()
    dates.sort()


    for date,a in zip(dates,axes.ravel()):
        
        Data = GetAndFilter(Chips,Barcode,date)
    
        Data['Eto'] = pd.cut(Data.Etoposide,cut_eto,labels=lab_eto)
      
            
        Data['Cis'] = pd.cut(Data.Cisplatin,cut_cis,labels=lab_cis)
       
        Data.viability = pd.to_numeric(Data.viability,errors='coerce')
        Data = Data.groupby(['Cis','Eto']).count()
        Data[Data.viability==0] = np.nan
        
        Data=Data.viability.unstack()
        
        C = Data.copy()
        C[(C<10)|(np.isnan(C))] = ''
        C = pd.DataFrame(C,dtype=str)
            
        sns.heatmap(Data.iloc[::-1],annot=C.iloc[::-1],ax=a,cmap='viridis',vmax=10,fmt='s',cbar_kws={'label': 'Number of replicates'})
    plt.show()
    plt.savefig("Replicates.svg")
    
#%% Heatmap of viability

    
def plot_heatmap(Chips,Barcode,cut_eto,lab_eto,cut_cis,lab_cis):
    
    fig,axes = plt.subplots(2,3,squeeze=False)
    dates = Chips.date.unique()
    dates.sort()

    
    
    for date,a in zip(dates,axes.ravel()):
        
        Data = GetAndFilter(Chips,Barcode,date)
        # discretize values into labels
        Data['Eto'] = pd.cut(Data.Etoposide,cut_eto,labels=lab_eto)
        # [3e-4,3e-3,4e-3,3e-2,.11,.3,1.3,4.8,40],\   #typical values of cut_eto and lab_eto
        # labels=['Ctrl','',1e-3,7e-2,.18,.75,2.69,8])
            
        Data['Cis'] = pd.cut(Data.Cisplatin,cut_cis,labels=lab_cis)
        # [1e-3,1e-2,1.01e-2,.055,.16,.5,1.6,5,16,50],\
        #labels=['Ctrl','',3e-2,.1,.3,1,3,10,30])
        
        Data.viability = pd.to_numeric(Data.viability,errors='coerce')
        Data = Data.groupby(['Cis','Eto']).median()
        
        Data=Data.viability.unstack()
        
        sns.heatmap(Data.iloc[::-1],ax=a,cmap='viridis',vmin=0,vmax=1,cbar_kws={'label': 'Viability'})
        a.set_title(day(date,dates))
    
    plt.show()
    


def Bliss(Chips,Barcode,cut_eto,lab_eto,cut_cis,lab_cis):
    
    fig,axes = plt.subplots(2,3)
    dates = Chips.date.unique()
    dates.sort()
    
    for date,a in zip(dates,axes.ravel()):
        
        Data = GetAndFilter(Chips,Barcode,date)
        # discretize values into labels
        Data['Eto'] = pd.cut(Data.Etoposide,cut_eto,labels=lab_eto)
        # [3e-4,3e-3,4e-3,3e-2,.11,.3,1.3,4.8,40],\   #typical values of cut_eto and lab_eto
        # labels=['Ctrl','',1e-3,7e-2,.18,.75,2.69,8])
            
        Data['Cis'] = pd.cut(Data.Cisplatin,cut_cis,labels=lab_cis)
        # [1e-3,1e-2,1.01e-2,.055,.16,.5,1.6,5,16,50],\
        #labels=['Ctrl','',3e-2,.1,.3,1,3,10,30])
        
        Data.viability = pd.to_numeric(Data.viability,errors='coerce')
        Data = Data.groupby(['Cis','Eto']).median()
        
        Data=Data.viability.unstack()
        DataBis = Data.copy()
        
        m,n = Data.shape
        
        i_00 = 1 - Data.iloc[0,0]  #change here to i_00 = 0 if you want to remove my approx
        # i_00 = 0
        
        for i in range(m) :
            for j in range(n):
                
                
                i_IJ = 1 - Data.iloc[i,j] - i_00
                i_0J = 1 - Data.iloc[0,j] - i_00
                i_I0 = 1 - Data.iloc[i,0] - i_00
                
                DataBis.iloc[i,j] = i_IJ - (i_I0+i_0J-i_I0*i_0J)
                
        e = .5
        sns.heatmap(DataBis.iloc[::-1],ax=a,cmap='vlag',vmin=-e,vmax=e,cbar_kws={'label': 'Synergy'})
        a.set_title(day(date,dates))




#%% Scatter of viability


def plot_scatter(Chips,Barcode):
    
    fig,axes = plt.subplots(2,3)
    dates = Chips.date.unique()
    dates.sort()
    
    for date,a in zip(dates,axes.ravel()):
        
        Data = GetAndFilter(Chips,Barcode,date)
      
        kargs={'x':"Etoposide",'y':"Cisplatin",'data':Data,\
                            'hue':Data.viability,'hue_norm':(0,1),\
                                'palette':'viridis','ax':a,'alpha':.4,\
                                'sizes':(20, 200),'size_norm':(25,100),'size':'radius'}
    
        g = sns.scatterplot(**kargs)
            
        a.set_xscale('log')
        a.set_yscale('log')
    
        if date!=dates[0]:g.legend(bbox_to_anchor= (10, 1))
        a.set_title(day(date,dates))
        
    
#%% Single drop viability


def plot_single(Chips,Barcode,eto0,cis0):
    
    """Plot wells were only one drug is present, to compare solo-drugs IC50"""
    
    dates = Chips.date.unique()
    dates.sort()
    
    palette = [(0.269944, 0.014625, 0.341379)]+sns.color_palette('viridis',n_colors=len(dates)-2)+[(0.974417, 0.90359, 0.130215)]
    
    
    fig,[a1,a2] = plt.subplots(1,2)
    
    
    DATA = GetAndFilter(Chips,Barcode)
    
    Data = DATA[(DATA.Etoposide<3e-3)&(DATA.Cisplatin>1e-2)]
    DataCtrl = DATA[(DATA.Etoposide<3e-3)&(DATA.Cisplatin<1e-2)]
    
    
    for date,col in zip(dates,palette):
        
        D = Data[Data.date==date]
        if date in dates[cis0:] :ic50=fit_sigm(D.viability,D.Cisplatin,a1,col,1)
        else:ic50=np.nan
        sns.scatterplot(data=D,x="Cisplatin",y="viability",\
                        color=col,ax=a1,alpha=.7,linewidth=0,\
                            label=f"D{(datetime.strptime(date,'%y%m%d')-datetime.strptime(dates[0],'%y%m%d')).days}   IC50:{np.round(ic50,2)}")
            
        
    a1.set_xscale('log')
    a1.set_ylim(-.1,1.1)
    a1.legend()
    
    Data = DATA[(DATA.Etoposide>3e-3)&(DATA.Cisplatin<1e-2)]
    for date,col in zip(dates,palette):
        D = Data[Data.date==date]
        if date in dates[eto0:] :ic50=fit_sigm(D.viability,D.Etoposide,a2,col,1)
        else:ic50=np.nan
        sns.scatterplot(data=D,x="Etoposide",y="viability",\
                        color=col,ax=a2,alpha=.7,linewidth=0,\
                            label=f"D{(datetime.strptime(date,'%y%m%d')-datetime.strptime(dates[0],'%y%m%d')).days}   IC50:{np.round(ic50,2)}")
        
    a2.set_xscale('log')
    a2.set_ylim(-.1,1.1)
    a2.legend()
    
    
     
    
#%% Synergy plot

def find_IC50(DATA,dates,eto0,cis0):
    
    """ Finding the IC50 in function of dates and drugs for one xp, 
    need first days of each drug (eto0 and cis0)"""
    
    IC_cis = []
    ViabmidCis = []
    LCis = []
    
    # Cisplatin IC50s
    
    Data = DATA[(DATA.Etoposide<3e-3)&(DATA.Cisplatin>1e-2)]
    for date in dates:
        D = Data[Data.date==date]
        if date in dates[cis0:] :
            popt=find_params(D.viability,D.Cisplatin)        
            x05 = popt[1]
            v05 = popt[0]/2+popt[-1]/2
            l05 = popt[0]/2
        else:
            x05,v05,l05 = np.nan,np.nan,np.nan
        
        IC_cis.append(x05)
        ViabmidCis.append(v05)    
        LCis.append(l05)
        
    
    Data = DATA[(DATA.Etoposide>3e-3)&(DATA.Cisplatin<1e-2)]
        
    IC_eto = []
    ViabmidEto= []
    LEto = []
    
    # Etoposide IC50s
    for date in dates:
        D = Data[Data.date==date]
        if date in dates[eto0:] :
            popt=find_params(D.viability,D.Etoposide)        
            x05 = popt[1]
            v05 = popt[0]/2+popt[-1]/2
            l05 = popt[0]/2
        else:
            x05,v05,l05 = np.nan,np.nan,np.nan
        
        IC_eto.append(x05)
        ViabmidEto.append(v05)
        LEto.append(l05)
        
        
    L = np.mean((LCis,LEto),0)
    return pd.DataFrame({'date':dates,'cis':IC_cis,'vcis':ViabmidCis,'eto':IC_eto,'veto':ViabmidEto,'L':L})




def synergy(Chips,Barcode,eto0,cis0):
    """Plot discretized viability (dead, living, intermediate) based on 
    sphero viab compared to max and min viab"""
    dates = Chips.date.unique()
    dates.sort()
    
    
    DATA = GetAndFilter(Chips,Barcode)
    
    IC50 = find_IC50(DATA,dates,eto0,cis0)
    
    fig,axes = plt.subplots(2,3)
  
    for date,a in zip(dates,axes.ravel()):
        
    
        Data = GetAndFilter(Chips,Barcode,date)
                
        ic,ie,vc,ve,L = IC50[IC50.date==date][['cis','eto','vcis','veto','L']].values[0]
        
        

        if not (np.isnan(ic) or np.isnan(ie)):
            
            Viab = np.mean((vc,ve))
            X = np.logspace(-2,np.log10(ie*(1-10**-2/ic)),100)
            Data['viab_lvl'] = pd.cut(Data.viability,[0,Viab-L/3,Viab+L/3,1],labels=['dead','intermediate','living'])
       
            kargs={'x':"Etoposide",'y':"Cisplatin",'data':Data[Data.viab_lvl=='intermediate'],\
                            'hue':'viab_lvl',\
                                'palette':'Set1','ax':a,'alpha':.4,\
                                'sizes':(20, 200),'size_norm':(25,100),'size':'radius'}
            Y = ic*(1-X/ie)
            a.plot(X,Y)
        
        
        else:
             kargs={'x':"Etoposide",'y':"Cisplatin",'data':Data,\
                            'hue':Data.viability,'hue_norm':(0,1),\
                                'palette':'viridis','ax':a,'alpha':.4,\
                                'sizes':(20, 200),'size_norm':(25,100),'size':'radius'}
            
        
        g = sns.scatterplot(**kargs)
        a.set_xscale('log')
        a.set_yscale('log')

        if date!=dates[0]:g.legend(bbox_to_anchor= (10, 1))
        a.set_title(day(date,dates))
    a.legend()
    
    
#%% FIC2 analysis


def FIC2(Chips,Barcode,eto0,cis0):
    
    """Return FIC2 for an expermiment (need first days of each drug) 
    for all days (if possible)"""
    
    fig,axes = plt.subplots(2,3)
    dates = Chips.date.unique()
    dates.sort()
    IC50 = pd.DataFrame({'date':[1,2,3,4,5],'eto':[11.3,2.07,0.93,.84,.83],'cis':[12.7,1.83,1.04,.42,.17]})
    FIC= []
    
    for i,(date,a) in enumerate(zip(dates,axes.ravel())):
        
    
        Data = GetAndFilter(Chips,Barcode,date)
        Data = Data[abs(Data.Cisplatin-Data.Etoposide)/Data.Cisplatin <= .9]
        Data['drug'] = np.sqrt(Data.Cisplatin*Data.Etoposide)
       
        kargs={'x':"drug",'y':"viability",'data':Data,\
                            'hue':Data.radius,\
                                'palette':'viridis','ax':a,'alpha':.4}
    
            
        a.set_xscale('log')
        if date in dates[max(eto0,cis0):] :
            ic50=fit_sigm(Data.viability,Data.drug,a)
            ie = IC50[IC50.date==i-eto0+1].eto.values[0]
            ic = IC50[IC50.date==i-cis0+1].cis.values[0]
            FIC.append(ic50/(.5*(ie+ic)))
            a.set_ylim((-.1,1.1))

        else:
            kargs={'x':"Etoposide",'y':"Cisplatin",'data':Data,\
                            'hue':Data.viability,'hue_norm':(0,1),\
                                'palette':'viridis','ax':a,'alpha':.4}
            a.set_yscale('log')
        g = sns.scatterplot(**kargs)

        if date!=dates[0]:g.legend(bbox_to_anchor= (10, 1))
        a.set_title(day(date,dates))
    print(FIC)
    
