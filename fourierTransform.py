# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:15:55 2019

@author: rafme
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
import seaborn as sns
from scipy import signal
import peakutils
from scipy.signal import savgol_filter


sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))




plt.close('all')
cv2.destroyAllWindows()

#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))





#filedir = easygui.diropenbox() + '\\'    
#filepath = filedir + 'movementIndex.txt'
#filepath = easygui.fileopenbox(default="something")

filepath = easygui.diropenbox()
dirlist = [i for i in os.listdir(filepath) if os.path.isdir(os.path.join(filepath,i)) and 'Spontaneous' in i]


for kk in range(len(dirlist)):
    plt.close('all')
    datafile = os.path.join(filepath,dirlist[kk],'movementIndex.txt')

    data = np.transpose(np.loadtxt(datafile, skiprows=1))

    
#    dataX = data[0][:int(fps*5)]
#    dataY = data[1][:int(fps*5)]
    
    dataX = data[0]
    dataY = data[1]

    sec = dataX[1]-dataX[0]
    fps = 1/sec
    
    
    savepath = os.path.join(filepath,dirlist[kk])
    
    
    ##### THE FFT ALGORITHM
    Fs = fps;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,1,Ts) # time vector
    
    n = len(dataY) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n))] # frequency range
    frq = frq-(n/T)/2
    dataY = signal.detrend(dataY)
    Y = np.fft.fft(np.asarray(dataY))/n # fft computing and normalization
    Y = np.fft.fftshift(Y)
    
    
    '''PLOTS'''
    fig = plt.figure(figsize=(12,8)) 
    ax = fig.gca()
    plt.plot(frq,abs(Y)) # plotting the spectrum
    tic = Fs/2
    ax.set_xlim([-tic,tic])
    #plt.xticks(np.arange(-tic,tic,1),fontsize=30)
    plt.yticks(fontsize=22)
    plt.xlabel('Frequency (Hz)',fontsize=34)
    plt.ylabel('Abs(FFT)',fontsize=34)
    plt.tight_layout()
    
    indexes = peakutils.indexes(abs(Y),thres=0.4)
    [plt.plot(frq[indexes[i]],abs(Y)[indexes[i]],'ro') for i in range(len(indexes))]
    
    fig.savefig(savepath+'\\FFT.png', dpi=500)
    fig.savefig(savepath+'\\FFT.svg',format='svg',dpi=1200)   
    
    with open(savepath+'\\FFT_freq.txt','w') as f:
        f.write('Frequency (Hz)\tAbs(FFT)\n')
        for i in indexes:
            f.write('%.2f\t%.6f\n' % (frq[i],abs(Y)[i]))
    
    with open(savepath+'\\FFT.txt','w') as f:
        f.write('Frequency (Hz)\tAbs(FFT)\n')
        for i in range(len(frq)):
            f.write('%.2f\t%.6f\n' % (frq[i],abs(Y)[i]))
    
    '''PLOT SMOOTHED'''
    
    fig2 = plt.figure(figsize=(12,8)) 
    ax = fig.gca()
    
    tic = Fs/2
    ax.set_xlim([-tic,tic])
    #plt.xticks(np.arange(-tic,tic,1),fontsize=30)
    plt.yticks(fontsize=22)
    plt.xlabel('Frequency (Hz)',fontsize=34)
    plt.ylabel('Abs(FFT)',fontsize=34)
    plt.tight_layout()
    
    smooth = savgol_filter(abs(Y), 3, 1)
    
    plt.plot(frq,smooth)
    
    indexes_smooth = peakutils.indexes(smooth,thres=0.3)
    
    [plt.plot(frq[indexes_smooth[i]],smooth[indexes_smooth[i]],'ro') for i in range(len(indexes_smooth))]
    
    fig2.savefig(savepath+'\\FFT_smooth.png', dpi=500)
    fig2.savefig(savepath+'\\FFT_smooth.svg',format='svg',dpi=1200)   
    
    with open(savepath+'\\FFT_smooth_freq.txt','w') as f:
        f.write('Frequency (Hz)\tAbs(FFT)\n')
        for i in indexes_smooth:
            f.write('%.2f\t%.6f\n' % (frq[i],smooth[i]))
    
    with open(savepath+'\\FFT_smooth.txt','w') as f:
        f.write('Frequency (Hz)\tAbs(FFT)\n')
        for i in range(len(frq)):
            f.write('%.2f\t%.6f\n' % (frq[i],smooth[i]))


    '''Analysis of number of contractions per min'''
    fig3 = plt.figure(figsize=(12,8)) 
    ax = fig.gca()
    
    #plt.xticks(np.arange(-tic,tic,1),fontsize=30)
    plt.yticks(fontsize=22)
    plt.ylabel('Movement Index (a.u.)',fontsize=34)
    plt.xlabel('Time (s)',fontsize=34)
    plt.tight_layout()
    
    peaks, _ = signal.find_peaks(dataY, height=0, distance=4)
    
    totalTime = dataX[-1]
    peakspermin = 60*len(peaks)/totalTime

    
    plt.plot(dataX,dataY)
    plt.plot(dataX[peaks], dataY[peaks], "rx",markersize=10, label=str(peakspermin)+' contractions\nper min')
    plt.plot(dataX[peaks], dataY[peaks], "r.",markersize=6)
    plt.legend(fontsize=24)
    
    fig3.savefig(savepath+'\\peaksPerMin.png', dpi=500)
    fig3.savefig(savepath+'\\peaksPerMin.svg',format='svg',dpi=1200)   

    with open(savepath+'\\peaksPerMin.txt','w') as f:
        f.write('Peaks per min:\t%.2f' % (peakspermin))



