"""
Color Tracker Project with OpenCV

    step 0: camera input
    step 1: preprocessing: CLAHE, blur
    step 2: masking: color thresholding, morphology
    step 3: contour detection and filtering
    step 4: polygon approximation
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
    """Color tracker functionality with processing done by the run command"""
    def __init__(self, mode: ThresholdMode, custom_color: CustomColor, 
                 capture_source: str | int, result_window: str):
        # color modes
        self.mode: ThresholdMode = mode
        self.custom_color = custom_color
        # capture source
        self.cap = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
        # windowing
        self.result_window = result_window
        cv2.namedWindow(self.result_window, cv2.WINDOW_NORMAL)
        # misc
        self.trail_points = deque(maxlen = 32)
        self.running = True
        # for fps count
        self.prev_ticks = cv2.getTickCount()

    def end(self) -> None:
        """release memory and close"""
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Essential loop function handling all image processing"""
        ret, frame = self.cap.read()
        if not ret: 
            return

        # preprocessing
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hsv = preprocess(frame_hsv)

        # thresholding
        mask = None
        match self.mode:
            case ThresholdMode.BLUE:
                mask = create_mask(frame_hsv,
                                    lower_hsv = (100, 120, 120),
                                    upper_hsv = (140, 255, 255))
            case ThresholdMode.YELLOW:
                mask = create_mask(frame_hsv, 
                                    lower_hsv = (15, 120, 120), 
                                    upper_hsv = (35, 255, 255))
            case ThresholdMode.RED:
                mask = create_mask(frame_hsv,
                                    lower_hsv = (170, 120, 120),
                                    upper_hsv = (10, 255, 255))
            case ThresholdMode.CUSTOM:
                mask = create_mask(frame_hsv,
                                    lower_hsv = tuple(self.custom_color.lower_hsv),
                                    upper_hsv = tuple(self.custom_color.upper_hsv))

        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

        # contour detection
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            for cnt in contours:
                # select large area contours
                if cv2.contourArea(cnt) > 500:
                    # epsilon constant for polygon accuracy and depth
                    epsilon: float = 0.04 * cv2.arcLength(cnt, closed = True)
                    approx = cv2.approxPolyDP(cnt, epsilon, closed = True)

                    cv2.drawContours(frame, [approx], 0, (255, 255, 0), 2)
                
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
                                 param = (frame_hsv, self.custom_color))

        # calculate fps
        ticks: int = cv2.getTickCount()
        fps: float = cv2.getTickFrequency() / (ticks - self.prev_ticks)
        self.prev_ticks = ticks
        cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # display windows
        cv2.imshow("Mask", mask if mask is not None
                                else np.zeros_like(frame[:, :, 0], dtype = np.uint8))
        cv2.imshow(self.result_window, frame)

def main():
    # create the color tracker
    color_tracker = ColorTracker(
        ThresholdMode.YELLOW,
        CustomColor(lower_hsv = (15, 120, 120), 
                    upper_hsv = (35, 255, 255),
                    h_tol = 28, s_tol = 70, v_tol = 120),
        capture_source = 0,
        result_window = "Result"
    )

    # run the loop
    while color_tracker.running:
        color_tracker.run()

    color_tracker.end()

if __name__ == "__main__":
    main()