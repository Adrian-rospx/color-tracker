"""
Color Tracker Project with OpenCV

    step 1: preprocessing: CLAHE, blur
    step 2: masking: color thresholding, morphology
    step 3: contour detection
    step 4: contour filtering
    step 5: bounding box, centroid display
    step 6: trail creation and display 
"""

import cv2
import numpy as np

from collections import deque

# user created modules
from masking import create_mask, preprocess
from colorutils import ThresholdMode, CustomColor, sample_color

class ColorTracker:
    # color modes
    mode: ThresholdMode
    custom_color: CustomColor
    # input source
    cap: cv2.VideoCapture
    # display
    result_window: str
    # misc
    trail_points: deque
    running: bool

    def __init__(self, mode: ThresholdMode, custom_color: CustomColor, 
                capture_source: str | int, result_window: str):
        self.mode = mode
        self.custom_color = custom_color

        self.cap = cv2.VideoCapture(capture_source)

        self.result_window = result_window
        cv2.namedWindow(self.result_window, cv2.WINDOW_NORMAL)

        self.trail_points = deque(maxlen = 32)
        self.running = True

    def end(self):
        """release memory and close"""
        self.cap.release()
        cv2.destroyAllWindows()

    # main loop
    def run(self):
        # convenience
        trail_points = self.trail_points

        ret, frame = self.cap.read()
        if not ret: 
            return

        # preprocessing
        res_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        res_hsv = preprocess(res_hsv)

        # thresholding
        mask = None
        match self.mode:
            case ThresholdMode.BLUE:
                mask = create_mask(res_hsv,
                                    lower_hsv = (100, 120, 120),
                                    upper_hsv = (140, 255, 255))
            case ThresholdMode.YELLOW:
                mask = create_mask(res_hsv, 
                                    lower_hsv = (15, 120, 120), 
                                    upper_hsv = (35, 255, 255))
            case ThresholdMode.RED:
                mask = create_mask(res_hsv,
                                    lower_hsv = (170, 120, 120),
                                    upper_hsv = (10, 255, 255))
            case ThresholdMode.CUSTOM:
                mask = create_mask(res_hsv,
                                    lower_hsv = tuple(self.custom_color.lower_hsv),
                                    upper_hsv = tuple(self.custom_color.upper_hsv))

        result = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)

        # contour detection
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        
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
            self.running = False    
        elif key == ord('n') or key == ord('N'):
            self.mode = ThresholdMode.NONE
        elif key == ord('b') or key == ord('B'):
            self.mode = ThresholdMode.BLUE
        elif key == ord('y') or key == ord('Y'):
            self.mode = ThresholdMode.YELLOW
        elif key == ord('r') or key == ord('R'):
            self.mode = ThresholdMode.RED
        elif key == ord('c') or key == ord('C'):
            self.mode = ThresholdMode.CUSTOM
            
        # mouse handling for custom color picker
        if self.mode == ThresholdMode.CUSTOM:
            cv2.setMouseCallback(self.result_window, sample_color, 
                                 param = (res_hsv, self.custom_color))

        # display windows
        cv2.imshow("Mask", mask if mask is not None
                                else np.zeros_like(frame[:, :, 0], dtype = np.uint8))
        cv2.imshow(self.result_window, result)

def main():
    # create the color tracker
    color_tracker = ColorTracker(
        ThresholdMode.YELLOW,
        CustomColor(lower_hsv = (15, 120, 120), 
                    upper_hsv = (35, 255, 255),
                    h_tol = 30, s_tol = 60, v_tol = 80),
        capture_source = 0,
        result_window = "Result"
    )

    # run the loop
    while color_tracker.running:
        color_tracker.run()

    color_tracker.end()

if __name__ == "__main__":
    main()