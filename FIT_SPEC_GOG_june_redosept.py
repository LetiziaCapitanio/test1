#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:47:27 2017

@author: letizia

###run FIT_SPEC_GOG2-test.py ../../TABLE_GOG/results2/gog_result.h5 ../DIBMOD/dib_model862.txt ../../TABLE_GOG/MODEL_RES_GOG/ ../../TABLE_GOG/MODEL_RES_GOG/lista_stellar_model
python FIT_SPEC_GOG_june_redosept.py //Volumes/Transcend/TABLE_GOG/RESULTS_Janvier/gog_result.h5 ../DIBMOD/dib_model862.txt //Volumes/Transcend/TABLE_GOG/results2/gog_result.h5 models_gog.txt

Program to fit the hdf5 spectra.
1. Reading hdf5 based on the openandread.py
2. Fitting based on the SophieFit_3.py

Estimating the rotation velocity for the stellar model
It assumes that we already know and pass the stellar velocity
./SophieFit.py sys.argv[1] sys.argv[2] sys.argv[3] sys.argv[4]
sys.argv[1]: tablegog.hdf5
sys.argv[2]: Dib model path
sys.argv[3]: Stellar models path
sys.argv[4]: stellar models list

Last modification: 
- cambiato vsini limite: fare per tutti la stessa cosa con bound [0.0001,100]
- cambiato bound per fare grado secondo delle calde
- salvare anche mediana del flusso per calcolare magnitudine apparente
- stelle calde: anche 7500 e 8500 K
- modello che viene da GOG e non dalla nostra griglia
- 5 marzo: vsini non libero, ma sono un "intorno" della vsini di inizio
- 7 marzo: normalizzazione col massimo dell'intervallo (cambiato)
- 2 giugno: per tutti i tipi stellari lo stesso fit. Normalizzazione.
  cambiato modo di calcolare la retta!
- 26 giugno: e poi cambio pure il calcolo di vsini, che tanto avremo il parametro stellare
  fittato bene e non capisco perche' dovremmo fingere il contrario
- settembre; si rifa' tutto e si salva benino

"""

import numpy as np
import sys
import h5py
import re
import datetime
import pandas as pd
#import re
#from astropy.io import ascii
from astropy.table import Table

#import os
import time
# Scientific modules
import scipy
from scipy import interpolate
from scipy import stats
import scipy.optimize as optimization
import scipy.integrate as integra
import scipy.constants # Module with physical magnitudes
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator

# Astro modules
from PyAstronomy.pyasl import rotBroad 
#from PyAstronomy.pyasl import broadGaussFast 
#from PyAstronomy.pyasl import crosscorrRV
from PyAstronomy.pyasl import instrBroadGaussFast 
#from PyAstronomy.pyasl import airtovac2

def splittami(x):
    return "*"+x.split('*')[1][:-1]

def find_pattern(x, a, b, c):
    for item in x:
        match = re.search(str(a),item[0])
        if match:
            match2 = re.search(str(b),item[0])
            if match2:
                match3 = re.search(str(c),item[0])
                if match3:
                    stellar_model_file  = item[0]
    return stellar_model_file     

def parabola(x,bb,b,c):
    return (bb*(x**2)+b*x +c )    
    
def mydibfit(x, a,bb, b, c): #HOT STAR
    global ydib   
    global myfluxmodel_forfit
    return (ydib**a)*(myfluxmodel_forfit)*(bb*(x**2)+b*x +c )

def mydibfit2(x, a, b, c, vsini,epsilon,d,radvel): #COLD WITH ROTATION           
    global ydib
    global myfluxmodel_forfit
    f = interpolate.interp1d(x, myfluxmodel_forfit, fill_value=1,\
                                 bounds_error=False)
    ymodel = f((x*(radvel)))#myflux
    yymodel = myrotbroad_filter(ymodel, epsilon, vsini)
    return (ydib**a)*(yymodel**d)*(b*x +c)

def mydibfit3(x, a, b, c,d,radvel): #COLD WITHOUT ROTATION
    global ydib 
    global myfluxmodel_forfit  
    f = interpolate.interp1d(x, myfluxmodel_forfit, fill_value=1,\
                                 bounds_error=False)
    ymodel = f((x*(radvel))) #myflux    
    return (ydib**a)*(ymodel**d)*(b*x +c)         

#-----------------------------------------

def myrdmodel(myfile):  
    tmode = Table.read(myfile, format='ascii.no_header', guess=False)
    lmode = tmode['col1'].data
    fmode = tmode['col2'].data
    return lmode, fmode

def myrotbroad_filter(y, epsilon, vsini):
    global mylambda_sel 
    return rotBroad(mylambda_sel, y, epsilon, vsini, edgeHandling="firstlast")

def calculate_EWDIB(xdibmod,ydibmod):#, mydibonlyfiterr2, condi):
    allones = np.ones(len(xdibmod))
    ewint = (integra.trapz(allones,xdibmod) \
          - integra.trapz(ydibmod,xdibmod))
#    ewinterr = 1.E3 * (integra.trapz(allones[condi],mylambdasel[condi]) \
#         - integra.trapz(mydibonlyfiterr2[condi],mylambdasel[condi]))
#    ewintB = (1 - np.min(mydibonlyfit2[condi])) * 1000.    
    return ewint#, ewinterr, ewintB    

def save_figures(mylambda_sel, myflux_sel, myfluxmodel_forfit, ydib, \
                 mymodelonlyfit, myresiduals, myalpha, myerralpha, \
                 EW_results, EW_results_error):
    global spectrum_ind

    xtext = 0.11
    ytext = 0.40
#    ytextoffset = 0.08
    myfontsize = 8
#    col1 = "SeaGreen"
    col2 = "deeppink"
    col3 = "green"
    col4 = "darkturquoise"
#    col5 = "limegreen"
    minorLocator = MultipleLocator(1.0)

    plt.figure(figsize=(8,5))
    
    ax = plt.axes([.1, .37, .85, .55])    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%+.2f'))
   # ax.xaxis.set_major_formatter(FormatStrFormatter('%+.2f'))
    nullfmt = NullFormatter()
    ax.xaxis.set_major_formatter(nullfmt)
    ax.xaxis.set_minor_locator(minorLocator)
    
    Yinterval= max(myflux_sel) - min(myflux_sel)
#    pltoffset1 = Yinterval*0.05
#    pltoffset2 = Yinterval*0.09
#    pltoffset3 = Yinterval*0.12
    
    # Data
    plt.plot(mylambda_sel,myflux_sel, 'k-', label='data',linewidth=2.5)
    # Fit results
    plt.plot(mylambda_sel, myfluxfit , '-', color=col2,linewidth=2.0, \
        label='Total fit')# + str(pltoffset1))    
    plt.plot(mylambda_sel, mydibonlyfit, '.-.', color=col3,\
        label='DIB mod fit')
    plt.plot(mylambda_sel, mymodelonlyfit , '-', color=col4,\
        linewidth=1.0, label='Stellar model fit' ) 
#    plt.plot(mylambda_sel,myflux_sel, 'k-', label='data',linewidth=2.5)
#    # Fit results
#    plt.plot(mylambda_sel, myfluxfit , '-', color=col2,linewidth=2.0, \
#        label='Total fit')# + str(pltoffset1))    
#    plt.plot(mylambda_sel, mydibonlyfit, '.-.', color=col3,\
#        label='DIB mod fit')
#    plt.plot(mylambda_sel, mymodelonlyfit, '-', color=col4,\
#        linewidth=1.0, label='Stellar model fit' ) 


    plt.xlim(xmin=mylambda_sel[0],xmax=mylambda_sel[-1])
    plt.ylim(ymin=min(myflux_sel)-0.15*Yinterval,ymax=max(myflux_sel)+0.15*Yinterval)
    plt.ylabel('Flux')
    
    labeldib = "Alpha = " + str("%.3f" %myalpha)+'\n' \
                 "AlphaErr = "+str("%.3f" %myerralpha )                                          
    plt.figtext(xtext,ytext, labeldib,color='r',\
                fontsize=myfontsize)
#    labelew = "EW = " + str("%.4f" %EW_results)+ '\n'\
#                 "EWErr = " ' '+str("%.4f" %EW_results_error )
#    plt.figtext(xtext,ytext+ytextoffset, labelew,color='r',\
#                fontsize=myfontsize)    
    plt.legend(loc=4)

    # Residuals
    ax = plt.axes([.1, .1, .85, .25])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%+.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%+.2f'))
    ax.xaxis.set_minor_locator(minorLocator)
#    plt.yticks(yticksresi)
    plt.plot(mylambda_sel,myresiduals/np.median(myflux_sel), color="k")
    plt.xlim(xmin=mylambda_sel[0],xmax=mylambda_sel[-1])
   # plt.ylim(ymin=-0.12,ymax=0.12)
    plt.axhline(y=0, color="Black") # for the DIB
    plt.xlabel('Wavelength (nm)')
    
#    plt.show()
    
    plt.savefig('/Users/letizia/Documents/THESIS2016/FIT_SIMU_GAIA/REDO_SEPTEMBRE/'+
                str(spectrum_ind)+"_Temp"+str(teffinput)+
                "_Log"+ str(logginput)+"_Fe_H"+
                str(fehinput)+"_Vsin"+ str(vsiniinput)+ 
                "_RadV"+str(radvelinput)+"_Avinput"+str(avinput) 
                +"_magVabs"+str(mean_v_absinput)+'redo_sept.png')
    plt.close()
    
    l="ok"
    return l  

grp_pattern = re.compile(r"\d+")  #che cazzo vuol dire? > casotto non risponde
a = datetime.datetime.now() 
print("Begin",a)

myinputerror = "Please, fix your input and launch again" 

##------Read input
try:
    h5File = sys.argv[1] # File with the correspondence h5file
    print("Myinput: ")
    print(h5File)
except:
    print(__doc__)
    sys.exit("sys1")
    
try:
    pathDIBmodel = sys.argv[2] # File with DIB model
    print("pathDIBmod:")
    print(pathDIBmodel)
except:
    print(__doc__)
    sys.exit("sys2")
    
try:
    pathmod = sys.argv[3] #path for the stellar model
    print("pathmod:")
    print(pathmod)
except:
    print(__doc__)
    sys.exit("sys3")
#    
try:
 #  print(sys.argv[4])  
 #  stellar_model_list = sys.argv[4] #stellar model list
   print("Stellar models list:")
#dspath feh teff logg
   g = pd.read_table('/Users/letizia/FIT/SIMULATION_FIT/results_gogandinput.txt', 
                     sep = ' ')
   dspathmodel = np.asarray(g['dspath'])
   teffmodel = np.asarray(g['teff'])
   loggmodel = np.asarray(g['logg'])
   fehmodel = np.asarray(g['fe_h'])
   magmodel = np.asarray(g['mean_abs_V'])
   vsinimodel = np.asarray(g['vsini'])
   avmodel = np.asarray(g['av'])
except:
   # print(sys.argv[4]) 
    print(__doc__)
    sys.exit("sys4")

#    

clight = scipy.constants.physical_constants['speed of light in vacuum'][0]/1.E3

h5 = h5py.File(h5File, "r")

#------------------------------------------------------------------------
#    We read the model for the  DIB
print("--------------------------------------------- ")
print(" Reading DIB model and fitting to a Gaussian  ")
print("             ", time.ctime())
print(" Also, creating output files and directories  ")
print("--------------------------------------------- ")


##### READ DIB MODEL
ddib = pd.read_table(pathDIBmodel, sep= ' ')
xdibmod = np.array(ddib['lamdibs8620'])/10 #nm
ydibmod = np.array(ddib['dibs8620Av1']) 
mydib = 862.0 #nm
myewdibmod=calculate_EWDIB(xdibmod,ydibmod) #nm

#-----------------------------------------

print("My DIB: " + str(mydib))

pathout = "fit" + str(mydib) + "/" 

#---------------------------------
h5File = sys.argv[1]
h5 = h5py.File(h5File, "r")

h5FileModel = sys.argv[3]
h5model = h5py.File(h5FileModel, "r")

myresol = 11200 #20000

#---------------------------------

ffile = "/Users/letizia/FIT/SIMULATION_FIT/"+\
"STATISTIC/GOGjanuary_redo2_totalgum.txt"

g = pd.read_csv(ffile)
dspathout = np.array(g['position'])
total_gum_number = np.array(g['total_gum'])
#cond_inGUM = total_gum_number != 0

#dspath_for_fit = dspathout.compress(cond_inGUM)
#---------------------------
#to do: change for the specific spectra sampling
#start = np.float("%.1f" % 846.0)
#step = np.float("%.3f" % 0.025)
#mylambda = np.linspace(start - step, (start - step) + 1160*step, 1160)
#
#---------------------------
print("Start Loop",datetime.datetime.now() - a)
 
myfontsize = 14
#
#ff = open("RESULTS_FIT_SIMULATIONJANVIER/REDO_JUNE/GOGjanuary_redo_septembre.txt",'w')
#ff.write("namestar position teff logg feh vsiniinput "+\
#          "avinput mean_abs_v flagmodel "+\
#         "EW[nm] EWerror "+\
#         "alpha erroralpha "+\
#         "m errorm "+\
#         "q errorq "+\
#         "delta errordelta "+\
#         "epsilon errepsilon "+
#         "vsini[kms-1] errvisini "+\
#         "radvel[kms-1] errradvel "+
#         "maxflux_sel "+\
#         "medianflux_sel "+\
#         "maxfluxTOT "+\
#         "medianfluxTOT "+\
#         "stddevresidual " +\
#         "mychi2 "+ \
#         "mean_abs_residual"+\
#         '\n')

#i = -1
kk = 0

#for grp in h5["/"]: #all groups
#    try:
#        grp != '__DATA_TYPES__' #not last one
#        if grp_pattern.search(grp) != None   :                 
#            for ds in h5[grp]: 
#               i = 1 + i # use enumerate in the for cycle, this is error prone
#               dspath = "/{}/{}".format(grp, ds) #hdf5 location

#for i,dspath in enumerate(dspathout):
for i,dspath in enumerate(dspathout[255912:255915]):    
               h5path = h5[dspath]
               start = h5path.attrs["wavelength0"]
               step = np.float("%.3f" % h5path.attrs["wavelength_step"])
               spectrum = np.array(h5path)
               spectrum_ind = splittami(str(h5path.attrs["source_extended_id"]))
               myflux = np.array(h5path)
               #input value
               teffinput = h5path.attrs["teff"]
               logginput = h5path.attrs["logg"]
               fehinput = h5path.attrs["fe_h"]
               vsiniinput = h5path.attrs["vsini"]
               radvelinput = h5path.attrs["radial_velocity"]
               mean_v_absinput = h5path.attrs["mean_abs_v"]
               avinput = h5path.attrs["av"]
               source_id = h5path.attrs["source_extended_id"]
               #####
               mylambda = np.linspace(start - step, (start - step) + \
                                       len(myflux)*step, len(myflux))
                
               try:
                   #read the corresponding model
                   try:
                       condT = teffmodel == teffinput
                       condL = loggmodel == logginput
                       condF = fehmodel == fehinput
                       condM = magmodel == min(magmodel.compress(condT&condL&condF))
                       condVSINI = vsinimodel == min(vsinimodel.compress(condT&condL&condF))
                       condAV = avmodel == min(avmodel.compress(condT&condL&condF))
                       
                       mydspathmodel = dspathmodel.compress(condT&condL&condF
                                                            &condM&condVSINI
                                                            &condAV)
                       myteffmodel = teffmodel.compress(condT&condL&condF
                                                            &condM&condVSINI
                                                            &condAV)
                       myloggmodel = loggmodel.compress(condT&condL&condF
                                                            &condM&condVSINI
                                                            &condAV)
                       myfehmodel = fehmodel.compress(condT&condL&condF
                                                            &condM&condVSINI
                                                            &condAV)
                       
                       
                       h5pathmodel = h5model[str(mydspathmodel[3])]
                       startM = h5pathmodel.attrs["wavelength0"]
                       stepM = np.float("%.3f" % h5pathmodel.attrs["wavelength_step"])
                       y1 = np.array(h5pathmodel)#/np.median(np.array(h5pathmodel))
                       x1 = np.arange(startM, startM+stepM*len(y1), step = stepM)
                        
                       f = interpolate.interp1d(x1, y1)
                       flagmodel = 1 
                   #in case of absence of model: right line
                   except:
                       x1 = np.linspace(start - step, (start - step) + 
                                        len(myflux)*step, len(myflux))
                       y1 = np.ones(len(x1))
                       
                       f = interpolate.interp1d(x1, y1)
                       kk = kk +1
                       flagmodel = 0
                       
                    
    
#                   cond_sel1 = mylambda > 861.0 #nm
#                   cond_sel2 = mylambda < 862.8 #nm
#                   cond_sel1 = mylambda > 859.0 
#                   cond_sel2 = mylambda < 864.0
                   cond_sel1 = mylambda > 860.6 #nm
                   cond_sel2 = mylambda < 863.6 #nm
#                   cond_sel1 = mylambda > 850.6 #nm
#                   cond_sel2 = mylambda < 870.6 #nm
#                    
                   cond_sel = cond_sel1 & cond_sel2
                   mylambda_sel = mylambda.compress(cond_sel)#[2:-2]
                   myflux_sel = myflux.compress(cond_sel)#[2:-2]
                   
                   #normalizzazione modello
                   x = x1.compress(cond_sel)
                   y = y1.compress(cond_sel)/np.max(y1.compress(cond_sel))
                        
                    #-------- We broaden the stellar spectrum resolution                   
                 #  fmodel_rot = instrBroadGaussFast(x,y,myresol)
                    
                    #-------- Resampling the model to the lambda of the data
#                   f = interpolate.interp1d(x, fmodel_rot, fill_value=1,\
#                                             bounds_error=False)
                   f = interpolate.interp1d(x, y, fill_value=1,\
                                             bounds_error=False)
                   myfluxmodel_forfit = f(mylambda_sel) #myflux
                    
                    # DIB INTERPOLATION
                   f = interpolate.interp1d(xdibmod, ydibmod, fill_value=1,\
                                             bounds_error=False)        
                   ydib = f(mylambda_sel)
                                       
            #----------------FIT-----------------------        
                   #separation for vsini = 0 and >0
                   #print("here")
                   # power dib
                   # a coefficient 
                   # m coefficient
                   # q coefficient
                   # vsini
                   # epsilon
                   # power model not
                   #radvel     
       
#                   m = (myflux_sel[-1] - myflux_sel[0])/(mylambda_sel[-1]-mylambda_sel[0])
#                   q = myflux_sel[-1]-m*mylambda_sel[-1]
#                   mmin = m-abs(m)*0.1
#                   mmax = m+abs(m)*0.1
#                   qmin = q-abs(q)*0.1
#                   qmax = q+abs(q)*0.1
                   m  = 0.0
                   q = np.max(myflux_sel)/np.max(myfluxmodel_forfit)
                   avmin = -0.1
                   avmax = 20.0
                   
                   p0 = [avinput,   #power DIB
                         m,     #mcoefficient
                         q,     #q coefficient
                         vsiniinput+1, #vsini
                         0.5,   #epsilon
                         1.0,   #power model
                         1.0]   #shift   #0.0] #radvel
                   
                   bounds = ([avmin,\
                              -np.inf, -np.inf, \
#                              mmin, \
#                              qmin,\
                              (vsiniinput+0.1),\
                              0.0,\
                              0.0, \
                              -2.0], \
                             [avmax,\
                               np.inf, np.inf, \
#                               mmax, \
#                               qmax,\
                               (vsiniinput+1),\
                               1,\
                               +np.inf,\
                               2.0])
                   popt, pcov = optimization.curve_fit(mydibfit2, \
                                        mylambda_sel,\
                                       myflux_sel,\
                                       p0=p0, bounds=bounds)
                   
                   #To compute one standard deviation errors on the parameters use 
                   # perr = np.sqrt(np.diag(pcov))
                   perr = np.sqrt(np.diag(pcov))
                   myalpha = popt[0]     #power dib
                   mym =  popt[1]        #m coefficient
                   myq = popt[2]         #q coefficien
                   myvsini = popt[3]     #vsini 
                   myepsilon = popt[4]   #epsilon Lambda Darkening                
                   mydelta = popt[5]     #power model
                   myradvel = (popt[6] -1)/clight  #radvel
                   
                   myfluxfit = mydibfit2(mylambda_sel,*popt)
                   mydibonlyfit = (ydib ** myalpha) *\
                                  (mylambda_sel*mym + myq)
                   mydibonlyfiterr = ydib ** perr[0]
                   
                   f = interpolate.interp1d(mylambda_sel, myfluxmodel_forfit, fill_value=1,\
                                     bounds_error=False)
                   ymodelFIT = f((mylambda_sel*popt[6]))
                   mymodelonlyfit = (myrotbroad_filter(ymodelFIT, \
                                                      myepsilon,\
                                                      myvsini)**mydelta)*\
                                                      (mylambda_sel*mym + myq)
                   myresiduals = myflux_sel - myfluxfit
                   mysigma = 0.
                  
                   myerralpha = perr[0]
                   myerrm =  perr[1]
                   myerrq= perr[2]              
                   myerrvsini = perr[3]
                   myerrepsilon = perr[4]
                   myerrdelta = perr[5]
                   myerrradvel = (perr[6]-1)/clight
                   
                   stddevmyresiduals = np.std(abs(myresiduals))
                   npoints = len(mylambda_sel)
                    # vtelu, alphatelu, sigmatelu, slopetelu.   
                   observed_values=myflux_sel
                   expected_values=myfluxfit
                   mychisquare = stats.chisquare(observed_values, f_exp=expected_values)       
                   mychisquarered = mychisquare[0] / (len(mylambda_sel) - len(popt) - 7) / \
                        stddevmyresiduals
                        
                   EW_results = myewdibmod*myalpha
                   EW_results_error =  myewdibmod*myerralpha   

               except:
                   popt = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                   myfluxfit = np.zeros(len(mylambda_sel))
                   mydibonlyfit = np.zeros(len(mylambda_sel))
                   mydibonlyfiterr = np.zeros(len(mylambda_sel))
                   mymodelonlyfit = np.zeros(len(mylambda_sel))
                   myresiduals = np.zeros(len(mylambda_sel))+9999
                   myalpha = 9999
                   mym =  9999
                   myq = 9999
                   myepsilon = 9999 #epsilon Lambda Darkening 
                   myvsini = 9999
                   mydelta = 9999
                   myradvel = 9999
                   myerralpha = 9999
                   myerrm =  9999
                   myerrq = 9999
                   myerrepsilon = 9999
                   myerrvsini = 9999
                   myerrdelta = 9999    
                   myerrradvel = 9999
                   stddevmyresiduals = 9999
                   mychisquare = 9999
                   mychisquarered = 9999
                   
                   EW_results = 9999
                   EW_results_error =  9999          
        #----------Save Results                                
#               ff.write(str(spectrum_ind)+' '+  \
#                         dspath  +' '+
#                         str(teffinput)+ ' '+\
#                         str(logginput) + ' '+\
#                         str(fehinput)+' '+\
#                         str(avinput)+' '+\
#                         str(vsiniinput)+' '+\
#                         str(mean_v_absinput)+' '+ str(flagmodel) +' '+\
#                         #fit results
#                         str("%.3f" %EW_results) + ' '+ str("%.3f" %EW_results_error)+' ' +\
#                         str("%.3f" %myalpha)+ ' '+str("%.3f" %myerralpha) + ' ' + \
#                         str("%.3f" %mym) + ' ' + str("%.3f" %myerrm) + ' '+ \
#                         str("%.3f" %myq) + ' ' + str("%.3f" %myerrq) + ' ' + \
#                         str("%.3f" %mydelta) + ' ' + str("%.3f" %myerrdelta) + ' ' + \
#                         str("%.3f" %myepsilon) + ' ' + str("%.3f" %myerrepsilon) + ' '+
#                         str("%.3f" %myvsini) + ' ' + str("%.3f" %myerrvsini) + ' '+  \
#                         str("%.3f" %myradvel) + ' ' + str("%.3f" %myerrradvel) + ' '+\
#                         #fit error
#                         #extimed SNR
#                         str("%.3f" %abs(np.max(myflux_sel))) + ' ' + \
#                         str("%.3f" %abs(np.median(myflux_sel))) + ' ' + \
#                         str("%.3f" %abs(np.max(myflux))) + ' ' + \
#                         str("%.3f" %abs(np.median(myflux))) + ' ' + \
#                         str("%.3f" %abs(np.median(myflux_sel/abs(myresiduals)))) + ' ' + \
#                         str("%.3f" %stddevmyresiduals) + ' ' + \
#                         str("%.3f" %mychisquarered) + ' ' + \
#                         str("%.3f" %np.mean(abs((myresiduals))))+\
#                          '\n') 
             
               if np.mod(i,1000)==0:    
                  print("             ", time.ctime(), str(spectrum_ind))            
                  print("Teff:", teffinput, "Logg", logginput, "FeH", fehinput, "sourceid", spectrum_ind)
                  print(p0)
                  print("dspath", dspath)
                  print("alpha:",myalpha,"error:", myerralpha, (myerralpha/myalpha)*100.0,'%')
                  print("emme:",mym,"error:", myerrm, (myerrm/mym)*100.0,'%')
                  print("qu:",myq,"error:", myerrq, (myerrq/myq)*100.0,'%')
                  print("delta:",mydelta,"error:", myerrdelta, (myerrdelta/mydelta)*100.0,'%')
                  print("vsini:",myvsini,"error:", myerrvsini, (myerrvsini/myvsini)*100.0,'%')
                  print("epsilon:",myepsilon,"error:", myerrepsilon, (myerrepsilon/myepsilon)*100.0,'%')
                  print("radvel_shift:",myradvel,"error:", myerrradvel, (myerrradvel/myradvel)*100.0,'%')
                  print("stddevresidual" , stddevmyresiduals, "mychi2", mychisquarered,\
                      "medianresidual", np.median(myresiduals))
                  
#               ll=save_figures(mylambda_sel, myflux_sel, myfluxmodel_forfit, ydib, 
#                                 mymodelonlyfit, myresiduals, myalpha,myerralpha, EW_results, EW_results_error)

               if np.mod(i,519)==0:
                    try:
                        print("models parameter",myteffmodel, myloggmodel, myfehmodel)
                    except:
                        print("noModel", kk)
                    print("delta:",myalpha,"error:", myerralpha, (myerralpha/myalpha)*100.0,'%')
               ll=save_figures(mylambda_sel, myflux_sel, myfluxmodel_forfit, ydib, \
                                 mymodelonlyfit, myresiduals, myalpha,myerralpha, EW_results, EW_results_error)
   
#                    plt.figure()
#                    plt.plot(mylambda_sel,myflux_sel/np.median(myflux_sel),'r')
#                    plt.plot(x,y,'b', label = 'model') 
#                    plt.legend()
#                    plt.savefig("test1.png")

#    except:
#        # except without the exception you want to catch is really bad
#        print("mmm")
#                                                  
#ff.close()        

b = datetime.datetime.now() 
c = b-a     

print("end",c)

