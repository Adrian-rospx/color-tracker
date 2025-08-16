import cv2
import numpy as np

import os
import sys

# check different source input
s: int = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source: cv2.VideoCapture = cv2.VideoCapture(s)

# create the window
window: str = "Color Tracker"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

# main loop
running: bool = True

while running:
    ret, frame = source.read()
    if not ret: 
        break

    

    # get the current key (and handle overflow)
    key: int = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q') or key == 27:
        running = False

    cv2.imshow(window, frame)

# release memory and close
source.release()
cv2.destroyWindow(window)