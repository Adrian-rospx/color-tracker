"""
Color filtering project using OpenCV
"""
import cv2
import numpy as np

import os
import sys

from enum import Enum

# check different source input
s: int = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
# start video caputure
source: cv2.VideoCapture = cv2.VideoCapture(s)

# create the window
window: str = "Color Tracker"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

result = None
running: bool = True

class ThresholdMode(Enum):
    # enums for color modes
    NONE = 0
    YELLOW = 1
mode = ThresholdMode.YELLOW

def yellowMask(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # masking function
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 80, 108])
    upper = np.array([40, 255, 255])

    return cv2.inRange(img_hsv, lower, upper)

# main loop
while running:
    ret, frame = source.read()
    if not ret: 
        break

    # process frame
    match mode:
        case ThresholdMode.NONE:
            result = frame
        case ThresholdMode.YELLOW:
            y_mask = yellowMask(frame)
            result = cv2.bitwise_and(frame, frame, mask = y_mask)

    # handle input
    key: int = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:
        # quit
        running = False    
    elif key == ord('n') or key == ord('N'):
        mode = ThresholdMode.NONE
    elif key == ord('y') or key == ord('Y'):
        mode = ThresholdMode.YELLOW

    # display
    cv2.imshow(window, result)

# release memory and close
source.release()
cv2.destroyWindow(window)