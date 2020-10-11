# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:25:35 2019

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
from detect_peaks import detect_peaks
import seaborn as sns

sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))


def calculateError(reference, image):
    
    pixelError = [np.abs(int(reference[i][j])-int(image[i][j])) for i in range(len(image)) for j in range(len(image[0]))]
    sumError = 0
    for i in range(0,len(pixelError)):
        sumError += pixelError[i]**2
    sumError = np.sqrt(sumError)

    return sumError


plt.close('all')
cv2.destroyAllWindows()

#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))


#The objective is specified to know the pixel size
#In the future, this will be asked, but in general it will be 10x
objective = 10

if objective == 10:
    sizeum = 1330.55
    sizepx = 2048
    


#The easygui function lets user select video instead of writing it by hand
filepath = easygui.fileopenbox(default="something")
#filepath = 'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\C2C12\\180317 20U D11 Moving pillars\\20 U D11 After stim.lif - stim min 120 pillar 1.avi'
filedir = os.path.splitext(filepath)[0]+'\\'
if not os.path.exists(filedir):
    os.makedirs(filedir)
dirlist = [ item for item in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, item)) ]

#fileName = filedir.split('.lif')[0].split('\\')[-1]
#fullFileName = filedir.split('.lif')[0].split('\\')[-1] + filedir.split('.lif')[1]
#print(fullFileName)

saveDir = filedir+'\\'

#Necessary to write videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Read video
video = cv2.VideoCapture(filepath)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))


# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
ok, initialFrame = video.read()

if not ok:
    print('Cannot read video file')
    sys.exit()
    

initialGray = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)

distance = list()



f = 0.4

frameResized = cv2.resize(initialFrame,(0,0),fx=f,fy=f)    
    
# Define an initial bounding box
bbox = (287, 23, 86, 320)

###bbox has four components:
#The first one is the x-coordinate of the upper left corner
#The second one is the y-coordinate of the upper left corner
#The third one is the x-coordinate of the lower right corner
#The four one is the y-coordinate of the lower right corner
 
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frameResized, False)

bbox = tuple([z/f for z in bbox]) #Resizes box to original file size
initialBbox = bbox
initialCrop = initialFrame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
initialCrop = cv2.cvtColor(initialCrop, cv2.COLOR_BGR2GRAY)
initialFrameResizedGray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)

count = 0
timeList = list()
centers = list()
errorCrop = list()
movementIndex = list()

save = True

pbar = tqdm(total=length) #For profress bar

while True:
    # Read a new frame
    pbar.update() #Update progress bar
    ok, frame = video.read()

    count += 1
    if not ok:
        break
    
    cropMovementIndex = frame[int(initialBbox[1]):int(initialBbox[1]+initialBbox[3]), 
                              int(initialBbox[0]):int(initialBbox[0]+initialBbox[2])]
    cropMovementIndex = cv2.cvtColor(cropMovementIndex, cv2.COLOR_BGR2GRAY)
    
    m = calculateError(initialCrop,cropMovementIndex)
    if count < 100:
        index = count
    else:
        index = 100
    if (m > 20*np.mean(movementIndex[-index:])):
        print('Found one outlier')
        movementIndex.append(movementIndex[-1])
    else:
        movementIndex.append(m)
    
    #Resize bbox
#    bbox = tuple(z*f for z in bbox)
    
    # Draw bounding box
    if save:
        frameResized = cv2.resize(frame,(0,0),fx=f,fy=f)  
        p1 = (int(bbox[0]*f), int(bbox[1]*f))
        p2 = (int(bbox[0]*f + bbox[2]*f), int(bbox[1]*f + bbox[3]*f))
        cv2.rectangle(frameResized, p1, p2, (0,0,255))
#        cv2.circle(frameResized,radius = 2,center=(int(centers[-1][0]/2),int(centers[-1][1]/2)),color=(255,0,0),thickness = -1)
        cv2.imwrite(saveDir+'frameBox.tif',frameResized)
        save = False
        
        

 
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

pbar.close() #Close progress bar

cv2.destroyAllWindows()
#out.release()
video.release()


sec = 1/fps
times = [i*sec for i in range(len(movementIndex)+1)]


'''IF THE FIRST IMAGE CORRESPONDED TO A PEAK, THIS CAN BE CORRECTED MANUALLY ONLY'''
correction = False

if correction:
    movementIndex = [max(movementIndex)*10 - movementIndex[i] for i in range(len(movementIndex))]
#Normalize movementIndex
movementIndex = [movementIndex[i]-min(movementIndex) for i in range(len(movementIndex))]

fig = plt.figure(figsize=(12,8))
plt.plot(times[1:],movementIndex)
plt.xlabel('Time [s]')
plt.ylabel('Arbitrary Units [a.u.]')
plt.tight_layout()
fig.savefig(saveDir+'movementIndex.png', dpi = 500)
fig.savefig(saveDir+'movementIndex.svg',format='svg',dpi=1200)   

f = open(saveDir+'movementIndex.txt', 'w') 
f.write('Time\tMovement Index (a.u.)\n')
for i in range(len(times[1:])):
    f.write("%.3f\t%.6f\n" % (times[i+1],movementIndex[i]))
f.close()











