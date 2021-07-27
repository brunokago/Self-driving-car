import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average

def make_coordinates(image,line_parameters):
    slope ,intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = ((y1 - intercept)/slope)
    x2 = ((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit =[]
    right_fit = []
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])
#canny used to show lines and edges
def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) # smoothening of image
    canny = cv2.Canny(blur,50,150) # caany method that returns the canny image
    return canny

#draw line in black image
def display_lines(image,lines):
    line_image = np.zeros_like(image)#make it same to the image
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),10)
    return line_image
#selecting area of interest 
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height), (550,250)]]) #setting the diagonals 
    mask = np.zeros_like(image) #
    cv2.fillPoly(mask,polygons,255)
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image) #loading the canny image to area of interest
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) #find lines in image
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,averaged_lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)#combine black image and lines with original line
cv2.imshow('result', line_image)
cv2.waitKey(0)