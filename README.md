# biobots-codes
 Different codes for tracking, force measurement

detect_peaks.py -> code for detecting peaks from Marcos Duarte (more information inside file)

imageDistance.py -> the user can select a cropped section of a video and calculate the "image distance" (or "movement index") throughout the whole duration of the video

postBending.py -> a video of a post being pulled by muscle tissue is uploaded and it is pre-processed by thresholding, Sobel edge detection, binarization and dilation to obtain the edges of the first frame (OpenCV). Then, the user selects a straight line perpendicular to the border of the post. The code then computes the displacement of the post along that line, using Gaussian approximations and Bresenham line algorithms. Pixels are converted to mm according to previously known information about the video conversion. Several videos or plots are produced: live tracking of the edge, displacement vs time, an animated plot of the edge detection. More information in: https://link.springer.com/chapter/10.1007/978-3-030-24741-6_18 and https://onlinelibrary.wiley.com/doi/full/10.1002/admt.201800631

forceMeasurement.py -> to be used after postBending.py. It simply uses the displacement information obtained from the previous code to obtain the force excerted.

fourierTransform.py -> it computes the Fourier transform and other metrics from "peak data" extracted by the previous codes.
