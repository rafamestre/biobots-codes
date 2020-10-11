# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:56:34 2017

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
import matplotlib.animation as animation
from scipy.signal import find_peaks


sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))


def animate(i):
    plt.cla()
    plt.plot(x,zi[i],label='Profile')
    plt.plot(x,gauss(x,avalue[i],meanValue[i],sigmas[i]),'r--',label='Gaussian fit')
    plt.vlines(x=meanValue[i],ymax=320, ymin=0,colors='g',linestyles='--')
    plt.xlabel('Pixels',fontsize=24)
    plt.ylabel('Pixel intensity',fontsize=24)
    plt.ylim(0,270)
    plt.tight_layout()



'''Function that defines the effects of clicking with the mouse
when drawing a line'''
def on_mouse(event, x, y, flags, params):
    #The scaled image and the scale value are passed as params
    im = params[0]
    image = im.copy()
    s = params[1]
    global line
    global lineFull
    global btn_down
    if event == cv2.EVENT_LBUTTONDOWN:
        line = [] #If we press the button, we reinitialize the line
        btn_down = True
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        line.append(sbox) 
        cv2.imshow("Sobel edge detection", image)
    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #this is just for line visualization
        cv2.line(image, (line[0][0],line[0][1]), (x, y), (256,0,0), 1)
        cv2.imshow("Sobel edge detection", image)
    elif event == cv2.EVENT_LBUTTONUP:
        #if you release the button, finish the line
        btn_down = False
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        line.append(ebox)
        cv2.line(image, (line[0][0],line[0][1]), (x, y), (256,0,0), 1)
        cv2.imshow("Sobel edge detection", image)
        lineFull = [[int(line[i][j]/(s/100)) for j in [0,1]] for i in [0,1]]


'''Function that's passed to the slider bar as a callback - does nothing'''
def interactiveEdge():
    pass

'''Gaussian function defined for fitting'''
def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


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
elif objective == 2.5:
    sizeum = 5322.2038
    sizepx = 2048
    

#The easygui function lets user select video instead of writing it by hand
if 'initialPath' in locals():
    default = initialPath
else:
    default = dn
filepath = easygui.fileopenbox(default=default)
initialPath = filepath.split(filepath.split('\\')[-1])[0]
#filepath = 'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\C2C12\\180317 20U D11 Moving pillars\\20 U D11 After stim.lif - stim min 120 pillar 1.avi'
filedir = os.path.splitext(filepath)[0]+'\\'
if not os.path.exists(filedir):
    os.makedirs(filedir)
dirlist = [ item for item in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, item)) ]


'''Uncomment this in case you can to puf a specific name to the file'''
#title = "File name"
#msg = "Enter file name"
#fieldValue = easygui.multenterbox(msg,title, ['File'])
## make sure that none of the fields was left blank
#while True:
#    if fieldValue == None: break
#    errmsg = ""
#    if fieldValue[0].strip() == "":
#        errmsg = errmsg + ('Add a valid file name')
#    else:
#        if len(dirlist) > 0 and fieldValue[0].strip() in dirlist:
#            overwrite = easygui.boolbox('File already exists. Do you want to overwrite it?', 
#                            'Attention', ['Yes', 'No'])
#            if not overwrite:
#                errmsg = errmsg + ('Add a non-existent file name')
#    if errmsg == "": break # no problems found
#    fieldValue = easygui.multenterbox(msg,title, ['File'])


saveDir = filedir+'\\'
#if not os.path.exists(saveDir):
#    os.makedirs(saveDir)




#Necessary to write videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# Read video
video = cv2.VideoCapture(filepath)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

#A scaling of 40% is applied to the image to be visible
f = 0.4

print(filepath)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
firstFrameBlack = False

ok, initialFrame = video.read()
if firstFrameBlack == True:
    ok, initialFrame = video.read()


if not ok:
    print('Cannot read video file')
    sys.exit()

#Resize frame
frameResized = cv2.resize(initialFrame,(0,0),fx=f,fy=f)    
#Set scale in percentage for slider bar
scale = int(f*100)
if 't' not in locals(): #Set threshold, otherweise, keep the last one
    t = 160 #Threshold for thresholding
kernel = np.ones((3,3),np.uint8) #kernel for dilation
if 'dilateIter' not in locals(): #Set dilate, otherwise, keep the last one
    dilateIter = 1
                
thres = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
(t, thres) = cv2.threshold(thres, t, 255, cv2.THRESH_BINARY_INV)
# perform Sobel edge detection in x and y dimensions
edgeX = cv2.Sobel(thres, cv2.CV_16S, 1, 0)
edgeY = cv2.Sobel(thres, cv2.CV_16S, 0, 1)
# convert back to 8-bit, unsigned numbers and combine edge images
edgeX = np.uint8(np.absolute(edgeX))
edgeY = np.uint8(np.absolute(edgeY))
#Binarize image
edges = cv2.bitwise_or(edgeX, edgeY)
#Dilate image
edges = cv2.dilate(edges,kernel,iterations = dilateIter)
#Resize edges to fit in screen
edgesResized = cv2.resize(edges,(0,0),fx=f,fy=f) 



#Values for the overlay
alpha = 0.5
gamma = 0
overlayFrame = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
cv2.addWeighted(edgesResized, alpha, overlayFrame, 1 - alpha,
		gamma, overlayFrame)

#First frame edges and slider bar for scale are shown
cv2.imshow('Sobel edge detection',edgesResized)
cv2.imshow('Overlayed image',overlayFrame)

cv2.namedWindow('Trackbars',cv2.WINDOW_AUTOSIZE )
cv2.createTrackbar('Scaling','Trackbars',scale,100,interactiveEdge)
cv2.createTrackbar('Threshold','Trackbars',int(t),300,interactiveEdge)
cv2.createTrackbar('Dilation level','Trackbars',int(dilateIter),5,interactiveEdge)

if 'line' not in locals():
    line = [] #Coordinates of the profiling line in the initial scaled image
if 'lineFull' not in locals():
    lineFull = [] #Coordinates of the profiling line in the full scale image
if 'lineResized' not in locals():
    lineResized = [] #Coordinates of the profiling line in the rescaled images

#A copy fo edgesResized will be sent to the mouse callback function to draw
#a different line every time
edgesResizedCopy = edgesResized.copy()
btn_down = False



while(True):
        
    cv2.imshow('Sobel edge detection',edgesResized)
    cv2.imshow('Overlayed image',overlayFrame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


    scale = cv2.getTrackbarPos('Scaling','Trackbars')
    t = cv2.getTrackbarPos('Threshold','Trackbars')
    dilateIter = cv2.getTrackbarPos('Dilation level','Trackbars')
#    (t, thres) = cv2.threshold(initialFrame, t, 255, cv2.THRESH_BINARY_INV)
    #edges = cv2.Canny(initialFrame,minVal,maxVal)
        
    thres = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    (t, thres) = cv2.threshold(thres, t, 255, cv2.THRESH_BINARY_INV)
    # perform Sobel edge detection in x and y dimensions
    edgeX = cv2.Sobel(thres, cv2.CV_16S, 1, 0)
    edgeY = cv2.Sobel(thres, cv2.CV_16S, 0, 1)
    # convert back to 8-bit, unsigned numbers and combine edge images
    edgeX = np.uint8(np.absolute(edgeX))
    edgeY = np.uint8(np.absolute(edgeY))
    edges = cv2.bitwise_or(edgeX, edgeY)
    edges = cv2.dilate(edges,kernel,iterations = dilateIter)
    
    edgesResized = cv2.resize(edges,(0,0),fx=scale/100,fy=scale/100)
    edgesResizedCopy = edgesResized.copy()
    cv2.setMouseCallback("Sobel edge detection", on_mouse, [edgesResizedCopy,scale])
    frameResized = cv2.resize(initialFrame,(0,0),fx=scale/100,fy=scale/100)
    overlayFrame = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
    cv2.addWeighted(edgesResized, alpha, overlayFrame, 1 - alpha,
                    gamma,overlayFrame)
    if len(lineFull) == 2: #If a line was drawn
        #Line is resized to be drawn in the resized edge frame
        lineResized = [[int(lineFull[i][j]*(scale/100)) for j in [0,1]] for i in [0,1]]
        cv2.line(edgesResized, (lineResized[0][0],lineResized[0][1]), 
                 (lineResized[1][0],lineResized[1][1]), (256,0,0), 1)
    
#ESC must be pressed to exit!

cv2.destroyAllWindows()

initialFrameCopy = initialFrame.copy()
#cv2.imwrite(saveDir+'frameNoLine.tif',initialFrameCopy)
#cv2.imwrite(saveDir+'edge.tif',edgesResized)
#cv2.line(initialFrameCopy,(lineFull[0][0],lineFull[0][1]), 
#                 (lineFull[1][0],lineFull[1][1]), (0,0,255), 4)
#cv2.imwrite(saveDir+'frameLine.tif',initialFrameCopy)


#Calcalate line size
lineSizepx = np.sqrt((lineFull[0][0]-lineFull[1][0])**2 + (lineFull[0][1]-lineFull[1][1])**2)
lineSizeum = lineSizepx*sizeum/sizepx


###line has four components:
#[0][0] is the x-coordinate of the upper left part
#[0][1] one is the y-coordinate of the upper left part
#[1][0] one is the x-coordinate of the lower right part
#[1][1] one is the y-coordinate of the lower right part
 
count = 0
profile = list()

num = 1000
x = np.linspace(lineFull[0][0],lineFull[1][0],num)
y = np.linspace(lineFull[0][1],lineFull[1][1],num)
zi = list()

#Applying Bresenham line's algorithm to get line in pixel positions
bresenhamList = list(bresenham(lineFull[0][0],lineFull[0][1],lineFull[1][0],lineFull[1][1]))
bresenhamList = [list(bresenhamList[k]) for k in range(len(bresenhamList))]
lineLength = len(bresenhamList)

#Get the profile for the initial frame, which was already opened
zi.append([edges[bresenhamList[px][1]][bresenhamList[px][0]] for px in range(len(bresenhamList))])
##VERY IMPORTANT: COORDINATES ARE SWITCHED IN IMAGES FROM OPENCV2
##Y-AXIS IS ACTUALLY X-AXIS

pbar = tqdm(total=length) #For profress bar

show = True
save = True

if save:
    #Necessary to write videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(saveDir+'liveTracking.avi', fourcc, fps, (int(width*f),int(height*f)))
    
           
for i in range(length-1):
    # Read a new frame
    pbar.update() #Update progress bar
    ok, frame = video.read()
    
    if not ok:
        break
    
    thres = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (t, thres) = cv2.threshold(thres, t, 255, cv2.THRESH_BINARY_INV)
    #edges = cv2.Canny(initialFrame,minVal,maxVal)
    # perform Sobel edge detection in x and y dimensions
    edgeX = cv2.Sobel(thres, cv2.CV_16S, 1, 0)
    edgeY = cv2.Sobel(thres, cv2.CV_16S, 0, 1)
    # convert back to 8-bit, unsigned numbers and combine edge images
    edgeX = np.uint8(np.absolute(edgeX))
    edgeY = np.uint8(np.absolute(edgeY))
    #Binarize image
    edges = cv2.bitwise_or(edgeX, edgeY)
    #Dilate image
    edges = cv2.dilate(edges,kernel,iterations = dilateIter)
    
    # Extract the values along the line, using Bresenham line's algorithm
    zi.append([edges[bresenhamList[px][1]][bresenhamList[px][0]] for px in range(len(bresenhamList))])
    ##VERY IMPORTANT: COORDINATES ARE SWITCHED IN IMAGES FROM OPENCV2
    ##Y-AXIS IS ACTUALLY X-AXIS
    

    # Display results only if necessary
    if show is True:
        overlayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.addWeighted(edges, alpha, overlayFrame, 1 - alpha,
                        gamma,overlayFrame)
        overlayFrame = cv2.cvtColor(overlayFrame, cv2.COLOR_GRAY2BGR)  
        cv2.line(overlayFrame,(lineFull[0][0],lineFull[0][1]), 
                     (lineFull[1][0],lineFull[1][1]), (0,0,255), 5)
        overlayFrameResized = cv2.resize(overlayFrame,(0,0),fx=f,fy=f)  
        cv2.imshow('Live update',overlayFrameResized)
        if save:
#            overlayFrameResized = cv2.cvtColor(overlayFrameResized, cv2.COLOR_GRAY2BGR)  
            out.write(overlayFrameResized)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            sys.exit()
            break
            
    
pbar.close() #Close progress bar
if save: 
    out.release()
video.release()

initialFrameCopy = initialFrame.copy()
cv2.line(initialFrameCopy,(lineFull[0][0],lineFull[0][1]), 
                 (lineFull[1][0],lineFull[1][1]), (0,0,255), 4)
cv2.imwrite(saveDir+'frameLine.tif',initialFrameCopy)

cv2.destroyAllWindows()

#Vector of times, frequency, period and ms per frame
#fps calculated from video import
fileName = filedir.split('.lif')[0].split('\\')[-1]
fullFileName = filedir.split('.lif')[0].split('\\')[-1] + filedir.split('.lif')[1]
    
freq = 1
period = 1/freq
nb = (length-1)/fps      
sec = 1/fps
times = [i*sec for i in range(length)]

#The mean value is calculated by fitting the edge to a gaussian function
#It's important that the images have dilation so that the gaussian can be properly fitted
meanValue = list()
sigmas = list()
avalue = list()
x = scipy.asarray(range(lineLength))


'''This part makes a video of the profiles.
Also it's adapted so that it takes the LAST peak in the profile
in case there were several edges of the image'''
example = True
snapshot = True

for i in range(len(zi)):
    y = zi[i]
#    mean = sum(x*y)/sum(y)                  
#    sigma = np.sqrt(sum(y*(x-mean)**2)/sum(y))      
    mean = find_peaks(zi[i])[0][-1] 
    try:
        popt,pcov = curve_fit(gauss,x,y,p0=[250,mean,0.2])
        if snapshot:
            figgg = plt.figure(figsize=(12,10))
            plt.plot(x,zi[i],label='Profile')
            plt.plot(x,gauss(x,*popt),'r--',label='Gaussian fit')
            plt.vlines(x=popt[1],ymax=320, ymin=0,colors='g',linestyles='--')
            plt.xlabel('Pixels',fontsize=24)
            plt.ylabel('Pixel intensity',fontsize=24)
            plt.legend()
            figgg.savefig(saveDir+'profile_example.png')
            figgg.savefig(saveDir+'profile_example.svg',format='svg',dpi=1200)  
            plt.close()
            snapshot = False
    except:
        pass
    meanValue.append(popt[1])
    sigmas.append(popt[2])
    avalue.append(popt[0])

if example:
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    figgg = plt.figure(figsize=(12,8))
    ani = matplotlib.animation.FuncAnimation(figgg, animate, frames=int(len(zi)), repeat=False)
    ani.save(saveDir+'profile_example.mp4', writer=writer)            


#The baselines of the contractions are calculated by finding exactly
#the contraction peaks
baselines = list()
#indeces = peakutils.indexes(np.asarray(meanValue), thres=0.6, min_dist=fps/2)
mph = np.abs(max(meanValue)-min(meanValue))/2 + min(meanValue)
indeces = detect_peaks(np.asarray(meanValue), mph=mph, mpd=period*2*fps/3)

maxPoints = [meanValue[i] for i in indeces]
indecesTime = [indeces*sec for indeces in indeces]

'''Old way of getting the baseline, a different one for each point
taking the median, but if the peak was too broad, this doesn't work'''
#for i in range(len(indeces)):
#    if indeces[i]-fps/4 < 0:
#        y = np.asarray(meanValue[:int(round(indeces[i]+fps/4))])
#    elif indeces[i] + fps/4 > len(meanValue):
#        y = np.asarray(meanValue[int(round(indeces[i]-fps/4)):])
#    else:
#        y = np.asarray(meanValue[int(round(indeces[i]-fps/4)):int(round(indeces[i]+fps/4))])
#    baselines.append(np.median(y))
#baselines = [np.mean(baselines)]*len(baselines)

medianBaseline = np.median(meanValue)
    
baselines = [medianBaseline]*len(indeces)

'''Two ways of calculating the baseline
First one is using peakutils.baseline
Might give rise to underestimation'''

baselines1 = peakutils.baseline(np.asarray(meanValue))
baselines2 = list()
baselines = list()

'''The other one is just taking the median of the data
Might give rise to overestimation'''

for i in range(len(indeces)):
    if (indeces[i]-fps/4) < 0:
        y = np.asarray(meanValue[:int(indeces[i]+fps/4)])
    elif (indeces[i]+fps/4) >= len(meanValue):
        y = np.asarray(meanValue[int(indeces[i]-fps/4):])
    else:
        y = np.asarray(meanValue[int(indeces[i]-fps/4):int(indeces[i]+fps/4)])
    baselines2.append(np.median(y))
'''Therefore I take as a baseline the average of both values'''

for i in range(len(indeces)):
    baselines.append(np.mean((baselines2[i],baselines1[indeces[i]])))

baselines = [np.mean(baselines1)]*len(baselines)
baselinesLong = [np.mean(baselines1)]*len(baselines1)
#Comparing the baseline to the peaks, we calculate the displacement in um      
increment = [(maxPoints[i]-baselines[i])*lineSizeum/lineLength for i in range(len(maxPoints))]

baselinesum = [baselines[i]*lineSizeum/lineLength for i in range(len(baselines))]

meanValueum = [meanValue[i]*lineSizeum/lineLength for i in range(len(meanValue))]
maxPointsum = [meanValueum[i] for i in indeces]

normalizedDisplacement = [meanValueum[i]-baselinesLong[i]*lineSizeum/lineLength for i in range(len(baselines1))]
normalizedMaxPoints = [maxPointsum[i]-baselinesum[i] for i in range(len(baselinesum))]

if firstFrameBlack == True:
    times = times[1:]

fig = plt.figure(figsize=(12,8))
plt.plot(times,meanValueum)
plt.xlabel('Time [s]')
plt.ylabel('Displacement [$\mu$m]')
plt.tight_layout()
fig.savefig(saveDir+'displacement.png', dpi = 500)
fig.savefig(saveDir+'displacement.svg',format='svg',dpi=1200) 
plt.plot(indecesTime,maxPointsum,'ro')
plt.plot(indecesTime,baselinesum,'g-')
fig.savefig(saveDir+'displacementPeaks.png', dpi = 500)
fig.savefig(saveDir+'displacementPeaks.svg',format='svg',dpi=1200) 


with open(saveDir+'displacement.txt', 'w') as file:
    file.write('Time\tDisplacement (um)\n')
    for i in range(len(times)):
        file.write("%.3f\t%.6f\n" % (times[i],meanValueum[i]))
    file.close()

fig = plt.figure(figsize=(12,8))
plt.plot(times, normalizedDisplacement)
plt.plot(indecesTime,normalizedMaxPoints,'ro')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [$\mu$m]')
plt.tight_layout()
fig.savefig(saveDir+'normalizedDisplacement.png', dpi = 500)
fig.savefig(saveDir+'normalizedDisplacement.svg',format='svg',dpi=1200) 

with open(saveDir+'normalizedDisplacement.txt', 'w') as file:
    file.write('Time\tDisplacement (um)\n')
    for i in range(len(times)):
        file.write("%.3f\t%.6f\n" % (times[i],normalizedDisplacement[i]))
    file.close()



with open(saveDir+'displacement.txt', 'w') as file:
    file.write('Time\tDisplacement (um)\n')
    for i in range(len(times)):
        file.write("%.3f\t%.6f\n" % (times[i],meanValueum[i]))
    file.close()



fig = plt.figure(figsize=(12,8))
plt.plot(indecesTime,increment,'o-')
plt.xlabel('Time [s]')
plt.ylabel('Absolute displacement [$\mu$m]')
plt.show()
plt.tight_layout()
fig.savefig(saveDir+'abstoluteDisplacement.png')
fig.savefig(saveDir+'abstoluteDisplacement.svg',format='svg',dpi=1200)   

with open(saveDir+'abstoluteDisplacement.txt', 'w') as file: 
    file.write('Time\tAbsolute displacement (um)\n')
    for i in range(len(indecesTime)):
        file.write("%.3f\t%.6f\n" % (indecesTime[i],increment[i]))
    file.close()



