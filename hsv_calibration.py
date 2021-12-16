import cv2
import numpy as np
import os
import sys

PATH_RGB = "./data/output/front_rgb"

def nothing(x):
    pass

def preprocess(img_data):
    hsv = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
    resized_hsv = cv2.resize(img_data, (144, 256), interpolation = cv2.INTER_AREA)

    return resized_hsv

# Load image
files = os.listdir(PATH_RGB)
file = "frame_12_09_2021_17_55_06_922604.png"
file2 = "frame_12_08_2021_21_06_52_270898.png"
#blue, green, red
image = cv2.imread(os.path.join(PATH_RGB, file))
image2 = cv2.imread(os.path.join(PATH_RGB, file2))

hsv = preprocess(image)
hsv2 = preprocess(image2)

# Create a window
cv2.namedWindow('hsv')
cv2.namedWindow('hsv2')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'hsv', 0, 255, nothing)
cv2.createTrackbar('SMin', 'hsv', 0, 255, nothing)
cv2.createTrackbar('VMin', 'hsv', 0, 255, nothing)
cv2.createTrackbar('HMax', 'hsv', 0, 255, nothing)
cv2.createTrackbar('SMax', 'hsv', 0, 255, nothing)
cv2.createTrackbar('VMax', 'hsv', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'hsv', 255)
cv2.setTrackbarPos('SMax', 'hsv', 255)
cv2.setTrackbarPos('VMax', 'hsv', 195)
cv2.setTrackbarPos('HMin', 'hsv', 179)
cv2.setTrackbarPos('SMin', 'hsv', 99)
cv2.setTrackbarPos('VMin', 'hsv', 0)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'hsv')
    sMin = cv2.getTrackbarPos('SMin', 'hsv')
    vMin = cv2.getTrackbarPos('VMin', 'hsv')
    hMax = cv2.getTrackbarPos('HMax', 'hsv')
    sMax = cv2.getTrackbarPos('SMax', 'hsv')
    vMax = cv2.getTrackbarPos('VMax', 'hsv')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(hsv, hsv, mask=mask)

    mask2 = cv2.inRange(hsv2, lower, upper)
    result2 = cv2.bitwise_and(hsv2, hsv2, mask=mask2)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('hsv', result)
    cv2.imshow('hsv2', result2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

