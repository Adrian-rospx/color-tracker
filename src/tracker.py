"""
Color Tracker Project with OpenCV

    step 1: masking: CLAHE, color thresholding, blurring, morphology
    step 3: contour detection
    step 4: filtering to largest area
    step 5: bounding box and center point display
    step 6: trail creation and display 
"""

import cv2
import numpy as np

import os
import sys

from collections import deque
from enum import Enum

# user created modules
from masking import createMask

class ThresholdMode(Enum):
    """Enum for color modes to be detected (none, blue, yellow, red)"""
    NONE = 0
    BLUE = 1
    YELLOW = 2
    RED = 3
mode = ThresholdMode.YELLOW

# start video capture
source: cv2.VideoCapture = cv2.VideoCapture(index = 0)
trail_points: deque = deque(maxlen = 32)
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
            mask = createMask(frame,
                                 lower_hsv = (100, 120, 120),
                                 upper_hsv = (140, 255, 255))
        case ThresholdMode.YELLOW:
            mask = createMask(frame, 
                                   lower_hsv = (15, 120, 120), 
                                   upper_hsv = (35, 255, 255))
        case ThresholdMode.RED:
            mask = createMask(frame,
                                   lower_hsv = (170, 120, 120),
                                   upper_hsv = (10, 255, 255))

    result = frame.copy()
    
    # find and draw contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        # select largest contour
        contour_max = max(contours, key = cv2.contourArea)

        if cv2.contourArea(contour_max) > 300:
            # draw the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour_max)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # draw center point
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour_max)
            center = (int(center_x), int(center_y))
            cv2.circle(result, center, 5, 
                       color = (0, 0, 255), thickness = -1)
    trail_points.appendleft(center)

    # draw trail
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue

        cv2.line(result, trail_points[i-1], trail_points[i],
                 color = (255, 255, 0), thickness = 2)
    
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