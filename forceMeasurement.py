# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:57:35 2018

@author: rmestre
"""


import os
import sys
import numpy as np
import cv2
import time
import glob
import matplotlib
from matplotlib import pyplot as plt
import easygui
import scipy.ndimage
from tqdm import tqdm
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy.optimize import curve_fit
import peakutils
from bresenham import bresenham
from detect_peaks import detect_peaks

matplotlib.rcdefaults() 

import seaborn as sns

sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))

font = {'fontname':'sans-serif'}
errorArgs = {'markeredgewidth':3.5,'elinewidth':3.5, 'capsize' : 5.5, 'linewidth' : 5}
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3


'''''''FORCE CALCULATION PART'''''''

def singF(y,x,E,w,L,h):
    '''This is for the calculation if the force is applied at the tip'''
    '''y -> displacement of the pillar in microns'''
    '''x -> height of the pillar where we record in mm'''
    '''E -> Young's modulus in mm'''
    '''w -> width of the pillar in mm'''
    '''L -> lateral size of the pillar in mm'''   
    '''h -> height of the pillar in mm'''
    #passing all units to m
    w = w/1000
    L = L/1000
    x = x/1000
    h = h/1000
    Iz = (w**3)*L/12
    force = list()
    if type(y) is  np.ndarray or type(y) is list:
        for i in range(len(y)):
            num = 1e-6*y[i]*6*E*Iz
            den = 3*h*x**2 - x**3
            force.append(num/den)
    elif type(y) is int or type(y) is float:
        num = 1e-6*y*6*E*Iz
        den = 3*h*x**2 - x**3
        force.append(num/den)
    return force


def singFmiddle(y,x,E,w,L,a,passive = False):
    '''This is for the calculation if the force is applied at a distance a'''
    '''y -> displacement of the pillar (in mm)'''
    '''x -> height of the pillar where measurement is made (in mm)'''
    '''E -> Young's modulus (in mm)'''
    '''w -> width of the pillar (in mm)'''
    '''L -> lateral size of the pillar (in mm)'''   
    '''a -> height where the force is applied (in mm)'''
    #passing all units to m
    w = w/1000
    L = L/1000
    a = a/1000
    Iz = (w**3)*L/12
    force = []
    if passive:
        if type(y) is np.ndarray or type(y) is list:
            if not (type(x) is np.ndarray or type(x) is list):
                print('Error: y and x are not both lists of values')
                sys.exit()
            elif not len(y) == len(x):
                print('Error: y and x are not the same size')
                sys.exit()
            for i in range(len(y)):
                x_m = x[i]/1000
                y_m= y[i]/1000
                num = y_m*6*E*Iz
                if x_m <= a:
                    den = 3*a*x_m**2 - x_m**3
                elif x_m > a:
                    den = 3*x_m*a**2 - a**3
                force.append(num/den)
        elif type(y) is int or type(y) is float:
            if type(x) is np.ndarray or type(x) is list:
                print('Error: y is a list and x is not')
                sys.exit()
            num = 1e-3*y*6*E*Iz
            x = x/1000
            if x <= a:
                den = 3*a*x**2 - x**3
            elif x > a:
                den = 3*x*a**2 - a**3
            force.append(num/den)           
    else:
        if type(y) is  np.ndarray or type(y) is list:
            if (type(x) is  np.ndarray or type(x) is list):
                print('Error: active force measurement does not allow different measurement heights')
                sys.exit()
            x = x/1000
            for i in range(len(y)):
                y_m= y[i]/1000
                num = y_m*6*E*Iz
                if x <= a:
                    den = 3*a*x**2 - x**3
                elif x > a:
                    den = 3*x*a**2 - a**3
                force.append(num/den)
        elif type(y) is not np.ndarray or type(y) is not list:
            num = 1e-3*y*6*E*Iz
            x = x/1000
            if x <= a:
                den = 3*a*x**2 - x**3
            elif x > a:
                den = 3*x*a**2 - a**3
            force.append(num/den)        

    return force
    
    
    

                            
            
def plotForces(filepath):
    '''THIS IS THE MAIN FUNCTION'''
    '''Goes through all the analyses in the folder and calculates the force'''
         
    plt.close('all')
    cv2.destroyAllWindows() 
    filedir = os.path.dirname(filepath)+'\\'  

    
    '''Code for calculating or reading passive force'''
    #Passive force in uN, default value
    passive = 0
    
    #Post parameters in mm, default values
    w = 0.60 #Width
    L = 2 #Lateral length
    h = 2.5 #Height
    a = 2 #Height where the tissue is pulling
    
    #Young's modulus default for 1:20
    E = 84870
    
    #Tissue diameter in m 
    '''ERASE'''
    diameter = 509.06953*1e-6
    diameter_std = 129.95*1e-6
    diameter_sem = 9.94*1e-6
    
    parametersFile = filedir+'parameters.csv'
    print(parametersFile)
    parameters = list()
    info = True
    try:
        f = open(parametersFile,'r')
        for line in f.readlines():
            parameters.append(line.split(';'))
        f.close()
    except FileNotFoundError:
#        print('Passive force information not found. Set to 0.')
        print('Young\'s modulus information not found.')
        print('Height information not found.')
#        print('Tissue thickness value not found. Set to 0 $\mu$m.')
        passiveInfo = False
        E = 100000
        diameter = 0
        diameter_std = 0
        diameter_sem = 0
        return
        
    if info:
        post1, post2, height1, height2, aux = ([] for i in range(5))
        for i in range(len(parameters)):
#            if parameters[i][0] == 'Post 1':
#                for j in range(1,len(parameters[i])):
#                    post1.append(float(parameters[i][j])/1000)
#            if parameters[i][0] == 'Post 2':
#                for j in range(1,len(parameters[i])):
#                    post2.append(float(parameters[i][j])/1000)            
            '''POST 1 and 2 ARE THE PASSIVE FORCES VALUES
            FOR THIS PROJECT THEY ARE NOT CONSIDERED
            BUT THE TEMPLATE OF THE PARAMETERS FILE HAS NOT BEEN CHANGED
            TO MINIMIZE THE CHANGES'''
            if parameters[i][0] == 'Height':
                for j in range(1,len(parameters[i])):
                    height1 = float(parameters[i][j])/1000
#            if parameters[i][0] == 'Height 2':
#                for j in range(1,len(parameters[i])):
#                    height2.append(float(parameters[i][j])/1000) 
            '''FOR NOW ONLY ONE POST IS BEING RECORDED.
            IF THE HEIGHT OF POST 2 IS 0, IT WILL ONLY ANALYZE ONE POST'''
#            if parameters[i][0] == 'Diameter':
#                diameter = float(parameters[i][1])*1e-6
#                diameter_std = float(parameters[i][2])*1e-6
#                diameter_sem = float(parameters[i][3])*1e-6
            '''DIAMETER FOR STRESS IS NOT CONSIDERED'''
            if parameters[i][0] == 'Modulus':
                E = float(parameters[i][1])
            '''THE OTHER TWO COMPONENTS OF THE MODULUS ARE THE STD AND SEM'''
            if parameters[i][0] == 'Width': #In um, so it's diviced by 1000 to mm
                w = (float(parameters[i][1])/1000)
            if parameters[i][0] == 'Lateral': #Lateral lengh in um, so it's diviced by 1000 to mm                    
                L = float(parameters[i][1])
                    
        '''PASSIVE FORCES ARE NOT CALCULATED FOR THIS PROJECT'''
#        passiveForce1 = singFmiddle(post1,height1,E,width[0],L,np.mean(height1),True)     
#        passiveForce2 = singFmiddle(post2,height2,E,width[1],L,np.mean(height2),True)  
#            
#        passive = np.mean((passiveForce1,passiveForce2))*1e6
#        passive_std = np.std((passiveForce1,passiveForce2))*1e6
#        passive_sem = passive_std/np.sqrt(len(passiveForce1)+(len(passiveForce2)))   
        
        #If there's a second post, the height is the average between both post's height
#        if height2[0] == 0: 
#            a = height1[0]
#        else:
#            a = np.mean((height1,height2))
        
#    elif not passiveInfo:
#        passive = 0
#        passive_std = 0
#        passive_sem = 0
#        E = 100000
#        diameter = 0
#        diameter_std = 0
#        diameter_sem = 0


    
#    #Area in mm^2
#    area = 1e6*(np.pi*diameter**2)/4
#    area_std = 1e6*(np.pi*diameter_std*diameter/2)
#    area_sem = 1e6*(np.pi*diameter_sem*diameter/2)
#    #For the area, we will always use SEM because it's the error of the mean
#    #For forces, we will use STD most of the time
#    
#    #Force in uN, area in mm^2, no need to change units, results in Pa
#    passiveStress = passive/area
#    passiveStress_std = stressError(passiveStress,area,passive_std,area_sem)
#    passiveStress_sem = stressError(passiveStress,area,passive_sem,area_sem)
    
#    with open(filedir+'..\\..\\dataPassiveForce.txt', 'w') as f:
#        f.write('Force (uN)\tForce std\tForce sem\tStress (Pa)\tSress std\tStress sem\n')
#        f.write("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (passive,
#                        passive_std,passive_sem,passiveStress,passiveStress_std,passiveStress_sem))
#        
#    
#    with open(filedir+'..\\..\\parametersUsedPassive.txt', 'w')  as f:
#        f.write('Young\'s modulus (kPa)\tWidth (mm)\tLength (mm)\tTissue height av. (mm)\tDiameter (mm)\tDiameter std\tDiameter sem\tArea (mm^2)\tArea std\tArea sem\n')
#        f.write("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n" % (E/1000,
#                        w,L,a,1e3*diameter,1e3*diameter_std,1e3*diameter_sem,area,area_std,area_sem))
    
    '''
    '''
    '''
    '''
    '''End code for calculating or reading passive force'''
    '''
    '''
    '''
    '''
    
    
    global data
    #Component 0 is time, component 1 is displacement
    data = np.transpose(np.loadtxt(filepath, skiprows=1))
    fps = 25
    sec = 1/fps
    period = 1
    


    
    #Singular force in N
    #Since force is linear with y(x), we can just calculate the force
    #of the increment
    disp = [1e-3*(data[1][i]) for i in range(len(data[1]))]
    force = singFmiddle(disp,height1,E,w,L,a)
    force = [force*1e6 for force in force]
    
    mph = np.abs(max(force)-min(force))/2 + min(force)
    indeces = detect_peaks(np.asarray(force), mph=mph, mpd=period*2*fps/3)
    
    maxPoints = [force[i] for i in indeces]
    indecesTime = [indeces*sec for indeces in indeces]


    
    with open(filedir+'forceTime.txt', 'w')  as f:    
        f.write('Force (uN)\tTime (s)\n')
        for i in range(len(force)):
            f.write("%.6f\t%.6f\n" % (force[i],data[0][i]))
    

    fig1 = plt.figure('Force in time',figsize=(12,8))
    plt.plot(data[0],force, linewidth = 2)
    plt.xlabel('Time (s)',fontsize=24)
    plt.ylabel('Force ($\mu$N)',fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    fig1.savefig(filedir+'forceTime.png',dpi=500)
    fig1.savefig(filedir+'forceTime.svg',format='svg',dpi=1200)  
    plt.plot(indecesTime,maxPoints,'ro')
    fig1.savefig(filedir+'forceTimePeaks.png', dpi = 500)
    fig1.savefig(filedir+'forceTimePeaks.svg',format='svg',dpi=1200) 

    meanForce = np.mean(maxPoints)
    sdtForce = np.std(maxPoints, ddof=1)
    semForce = sdtForce/np.sqrt(len(maxPoints))
    
    with open(filedir+'averageForce.txt', 'w')  as f:    
        f.write('Mean force(uN)\tSTD force (uN)\tSEM force (uN)\n')
        f.write("%.6f\t%.6f\t%.6f\n" % (meanForce,sdtForce,semForce))
    
    with open(filedir+'forcePeaks.txt', 'w')  as f:    
        f.write('Force(uN)\tTime (s)\n')
        for i in range(len(indecesTime)):
            f.write("%.6f\t%.6f\n" % (maxPoints[i],indecesTime[i]))

def plotForcesInTime(filepath):
    '''Creates the Analysis folder and plots all the forces in terms of time'''
    
    print(filepath)
    plt.close('all')
    cv2.destroyAllWindows() 
    filedir = os.path.dirname(filepath)+'\\'  


    if not os.path.exists(filedir+'Analysis'):
        os.makedirs(filedir+'Analysis')
    
    force,force_std,force_sem,stress,stress_std,stress_sem,area,area_std,area_sem,post,minute = ([] for i in range(11))


    dirlist = [ item.lower() for item in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, item)) ]

    for i in range(len(dirlist)):
        #The number of the post and the minute are taken automatically
        if (('pillar' in dirlist[i]) or (('post' or 'Post') in dirlist[i]) and 'min' in dirlist[i]):
            if ('After' or 'after' not in dirlist[i]):
                if 'pillar' in dirlist[i]:
                    positionP = dirlist[i].index('pillar')
                    postNb = dirlist[i][positionP+7] #The number of the pillar is taken
                    postNb = int(postNb)
                elif 'post' in dirlist[i]:
                    positionP = dirlist[i].index('post')
                    postNb = dirlist[i][positionP+5]
                    postNb = int(postNb)
            
            positionM = dirlist[i].index('min')
            if len(dirlist[i]) == positionM + 5:
                minNb = dirlist[i][positionM+4]
            elif len(dirlist[i]) == positionM + 6:
                minNb = dirlist[i][positionM+4:positionM+6]
            else:
                minNb = dirlist[i][positionM+4:positionM+7]
            if not minNb.isdigit():
                minNb = minNb[:-1]
                if not minNb.isdigit():
                    minNb = minNb[:-2]
            minNb = int(minNb)
                
            insideDir = filedir+dirlist[i]+'\\'
            insideDirlist = [ item for item in os.listdir(insideDir) if os.path.isdir(os.path.join(insideDir, item)) ]
            
            if len(insideDirlist) > 1:
                msg ="The folder \"" + dirlist[i] + "\" contains several analyses. Which one do you want to use?"
                title = "Choose analysis"
                choices = insideDirlist
                chosenFolder = easygui.choicebox(msg, title, choices)
            else:
                chosenFolder = insideDirlist[0]
            
            insideDir = insideDir + chosenFolder+'\\'
            insideFiles = os.listdir(insideDir)
            if "dataForce.txt" in insideFiles:
                positionFile = insideFiles.index('dataForce.txt')
                data = np.loadtxt(insideDir+insideFiles[positionFile], skiprows=1)
                force.append(data[0])
                force_std.append(data[1])
                force_sem.append(data[2])
                stress.append(data[3])
                stress_std.append(data[4])
                stress_sem.append(data[5])
                minute.append(minNb)
                post.append(postNb)
            
    indicesPost1 = [i for i, x in enumerate(post) if x == 1]
    indicesPost2 = [i for i, x in enumerate(post) if x == 2]
    indicesPosts = (indicesPost1,indicesPost2)
    
    colorPlot = 'b'
    colorPlot2 = 'orange'
    colorPlot3 = [colorPlot,colorPlot2]
    
    pairsIndices = list()
        
    for i in (0,1):
        
        if len(indicesPosts[i]) is not 0:
            minutes = [minute[k] for k in indicesPosts[i]]
            sortedMinuteIndeces = np.argsort(minutes)
            sortedMinute = [minutes[k] for k in sortedMinuteIndeces]
            sortedIndices = [indicesPosts[i][k] for k in sortedMinuteIndeces]
            pairsIndices.append(sortedIndices)
            sortedForce = [force[k] for k in sortedIndices]
            sortedForce_std = [force_std[k] for k in sortedIndices]
            sortedForce_sem = [force_sem[k] for k in sortedIndices]
            sortedStress = [stress[k] for k in sortedIndices]
            sortedStress_std = [stress_std[k] for k in sortedIndices]
            sortedStress_sem = [stress_sem[k] for k in sortedIndices]

            
            fig1 = plt.figure(i)
            plt.errorbar(sortedMinute,sortedForce, yerr=sortedForce_sem, 
                         markersize='0',**errorArgs)
            plt.xlabel('Time (min)',fontsize=22)
            plt.ylabel('Force ($\mu$N)',fontsize=22)
            plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            fig1.savefig(filedir+'\\Analysis\\forceInTime_P'+str(i+1)+'.png')
            fig1.savefig(filedir+'\\Analysis\\forceInTime_P'+str(i+1)+'.svg',format='svg',dpi=1200) 
            plt.close(fig1)            
            
            f = open(filedir+'Analysis\\forceInTime_P'+str(i+1)+'.txt', 'w') 
            f.write('Time (min)\tForce (uN)\tForce std\tForce sem\n')
            for k in range(len(sortedForce)):
                f.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (sortedMinute[k],sortedForce[k],
                                                      sortedForce_std[k],sortedForce_sem[k]))
            f.close()
            
            fig2 = plt.figure(i+10)
            plt.errorbar(sortedMinute,sortedStress, yerr=sortedStress_sem,
                         markersize='0',**errorArgs)
            plt.xlabel('Time (min)',fontsize=22)
            plt.ylabel('Stress (Pa)',fontsize=22)
            plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            
            fig2.savefig(filedir+'\\Analysis\\stressInTime_P'+str(i+1)+'.png')
            fig2.savefig(filedir+'\\Analysis\\stressInTime_P'+str(i+1)+'.svg',format='svg',dpi=1200) 
            plt.close(fig2)
            
            f = open(filedir+'Analysis\\stressInTime_P'+str(i+1)+'.txt', 'w') 
            f.write('Time (min)\tStress (Pa)\tStress std\tStress sem\n')
            for k in range(len(sortedStress)):
                f.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (sortedMinute[k],sortedStress[k],
                                                      sortedStress_std[k],sortedStress_sem[k]))
            f.close()
            
            fig3 = plt.figure(100)
            plt.errorbar(sortedMinute,sortedForce, yerr=sortedForce_sem,
                         markersize='0',**errorArgs,
                         label = 'Post '+str(i+1))
            plt.xlabel('Time (min)',fontsize=22)
            plt.ylabel('Force ($\mu$N)',fontsize=22)
            plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            
            fig4 = plt.figure(110)
            plt.errorbar(sortedMinute,sortedStress, yerr=sortedStress_sem,
                         markersize='0',**errorArgs,
                         label = 'Post '+str(i+1))
            plt.xlabel('Time (min)',fontsize=22)
            plt.ylabel('Stress (Pa)',fontsize=22)
            plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()

            
    plt.figure(100)
    plt.legend()
    ax1 = plt.gca()
    # get handles
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax1.legend(handles, labels, loc='upper left',numpoints=1,fontsize=16)
    plt.figure(110)
    plt.legend()
    ax1 = plt.gca()
    # get handles
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax1.legend(handles, labels, loc='upper left',numpoints=1,fontsize=16)
    
    

    forceAv = [(force[pairsIndices[0][k]]+force[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]
    forceAv_std = [(force_std[pairsIndices[0][k]]+force_std[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]
    forceAv_sem = [(force_sem[pairsIndices[0][k]]+force_sem[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]
    
    stressAv = [(stress[pairsIndices[0][k]]+stress[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]
    stressAv_std = [(stress_std[pairsIndices[0][k]]+stress_std[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]
    stressAv_sem = [(stress_sem[pairsIndices[0][k]]+stress_sem[pairsIndices[1][k]])/2 for k in range(len(pairsIndices[0]))]

    fig5 = plt.figure(200)
    plt.errorbar(sortedMinute,forceAv, yerr=forceAv_sem,
                         markersize='0',**errorArgs)
    plt.xlabel('Time (min)',fontsize=18)
    plt.ylabel('Force ($\mu$N)',fontsize=18)
    plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    f = open(filedir+'Analysis\\averageForceInTime.txt', 'w') 
    f.write('Time (min)\tForce (uN)\tForce std\tForce sem\n')
    for k in range(len(forceAv)):
        f.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (sortedMinute[k],forceAv[k],
                                              forceAv_std[k],forceAv_sem[k]))
    f.close()
    
    fig6 = plt.figure(300)
    plt.errorbar(sortedMinute,stressAv, yerr=stressAv_sem,
                         markersize='0',**errorArgs)
    plt.xlabel('Time (min)',fontsize=18)
    plt.ylabel('Stress (Pa)',fontsize=18)
    plt.xlim([sortedMinute[0]-5,sortedMinute[-1]+5])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    f = open(filedir+'Analysis\\averageStressInTime.txt', 'w') 
    f.write('Time (min)\tStress (Pa)\tStress std\tForce sem\n')
    for k in range(len(stressAv)):
        f.write("%.6f\t%.6f\t%.6f\t%.6f\n" % (sortedMinute[k],stressAv[k],
                                              stressAv_std[k],stressAv_sem[k]))
    f.close()
    
    

    fig3.savefig(filedir+'\\Analysis\\forceInTime_both.png')
    fig3.savefig(filedir+'\\Analysis\\forceInTime_both.svg',format='svg',dpi=1200)     

    fig4.savefig(filedir+'\\Analysis\\stressInTime_both.png')
    fig4.savefig(filedir+'\\Analysis\\stressInTime_both.svg',format='svg',dpi=1200)     

    fig5.savefig(filedir+'\\Analysis\\averageForceInTime.png')
    fig5.savefig(filedir+'\\Analysis\\averageForceInTime.svg',format='svg',dpi=1200)  

    fig6.savefig(filedir+'\\Analysis\\averageStressInTime.png')
    fig6.savefig(filedir+'\\Analysis\\averageStressInTime.svg',format='svg',dpi=1200)  




#compareFreq(r'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\3D BioBots\\Stimulations',10,[0.1,1,5,10])
#sys.exit()
#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))

#filep = 'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\C2C12\\180320 20U D14 Moving pillars\\while2.lif - stim pillar 2 min 120\\1\\displacement.txt'
#plotForces(filep)
#sys.exit()

allFiles = True

if allFiles:
    
    filedir = easygui.diropenbox() + '\\'
    folders = os.listdir(filedir)
    for i in range(len(folders)):
        filep = filedir + folders[i] + '\\normalizedDisplacement.txt'
        if os.path.exists(filep):
            plotForces(filep)
    
else:
    filedir = easygui.diropenbox() + '\\'    
    filep = filedir + 'normalizedDisplacement.txt'
    plotForces(filep)

sys.exit()





#The easygui function lets user select file instead of writing it by hand
filedir = easygui.diropenbox() + '\\'
#List of files within the selected directory
dirlist = [ item for item in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, item)) ]

for directory in dirlist:
    #Whole directory of each file
    filedir2 = filedir + directory + '\\'
    #Each video directory could have more than one analysis
    dirlist2 = [ item for item in os.listdir(filedir2) if os.path.isdir(os.path.join(filedir2, item)) ]
    for directory2 in dirlist2:
        #Directory of each analysis
        filedir3 = filedir2 + '\\' + directory2 + '\\'
        #List of all files within analysis folder
        files = os.listdir(filedir3)
        if 'normalizedDisplacement.txt' in files:
            #Use only the ones called displacement
            position = os.listdir(filedir3).index('normalizedDisplacement.txt')
            #Construct complete filepath
            filepath = filedir3 + files[position]
            #Calculate force
            plotForces(filepath, directory)
                

plotForcesInTime(filedir)



plt.close('all')



