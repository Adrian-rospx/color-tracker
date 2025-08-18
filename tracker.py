"""
Color Tracker Project with OpenCV

    step 1: color masking
    step 2: morphology
    step 3: contour detection
    step 4: bounding box display
"""
import cv2
import numpy as np

import os
import sys

from enum import Enum

class ThresholdMode(Enum):
    # enums for color modes
    NONE = 0
    BLUE = 1
    YELLOW = 2
    RED = 3
mode = ThresholdMode.YELLOW

def makeColorMask(img: cv2.typing.MatLike, lower_hsv: tuple[int], upper_hsv: tuple[int]) -> cv2.typing.MatLike:
    # masking function
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return cv2.inRange(img_hsv, lower_hsv, upper_hsv)

# start video capture
source: cv2.VideoCapture = cv2.VideoCapture(index = 0)
running: bool = True

# main loop
while running:
    ret, frame = source.read()
    if not ret: 
        break

    # color filtering
    mask = None
    match mode:
        case ThresholdMode.BLUE:
            mask = makeColorMask(frame,
                                 lower_hsv = (100, 120, 90),
                                 upper_hsv = (140, 255, 255))
        case ThresholdMode.YELLOW:
            mask = makeColorMask(frame, 
                                   lower_hsv = (15, 120, 90), 
                                   upper_hsv = (35, 255, 255))
        case ThresholdMode.RED:
            mask1 = makeColorMask(frame,
                                   lower_hsv = (170, 120, 90),
                                   upper_hsv = (180, 255, 255))
            mask2 = makeColorMask(frame,
                                   lower_hsv = (0, 120, 90),
                                   upper_hsv = (10, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
            
    # morphology for denoising
    if mode is not ThresholdMode.NONE:
        kernel = np.ones((12,12), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    result = frame

    # find and draw contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(result, contours, -1, color = (255, 255, 255), thickness = 2)

    # display contour bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # bounding box
        cv2.rectangle(result, 
                      (x, y), (x + w, y + h), 
                      color = (255, 255, 255), thickness = 2)

    # handle input
    key: int = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:
        # quit
        running = False    
    elif key == ord('n') or key == ord('N'):
        mode = ThresholdMode.NONE
    elif key == ord('b') or key == ord('B'):
        mode = ThresholdMode.BLUE
    elif key == ord('y') or key == ord('Y'):
        mode = ThresholdMode.YELLOW
    elif key == ord('r') or key == ord('R'):
        mode = ThresholdMode.RED

    # display windows
    cv2.imshow("Mask", mask if mask is not None
                            else np.zeros_like(frame[:, :, 0], dtype = np.uint8))
    cv2.imshow("Result", result)

# release memory and close
source.release()
cv2.destroyAllWindows()