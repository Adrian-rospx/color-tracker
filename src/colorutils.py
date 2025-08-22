"""Utility functions and objects:

- ThresholdMode enum
- CustomColor class for easier conversion
- sample_color for mouse click callback
"""

import numpy as np
import cv2

from enum import Enum
    
class ThresholdMode(Enum):
    """Enum for color modes to be detected (none, blue, yellow, red)"""
    NONE = 0
    BLUE = 1
    YELLOW = 2
    RED = 3
    CUSTOM = 4

class CustomColor:
    """Class handling custom color assignment, 
    setting tolerances and easing updates
    """
    lower_hsv: list[int]
    upper_hsv: list[int]

    h_tol: int
    s_tol: int
    v_tol: int

    def __init__(self, lower_hsv: list[int], upper_hsv: list[int],
                 h_tol: int, s_tol: int, v_tol):
        self.lower_hsv = list(lower_hsv)
        self.upper_hsv = list(upper_hsv)
        self.h_tol = h_tol
        self.s_tol = s_tol
        self.v_tol = v_tol

    def update(self, hsv: tuple[int]):
        """Update the lower and upper hsv based on 
        set tolerances with the new hsv value
        """
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        self.lower_hsv[0] = (h - self.h_tol//2 + 180) % 180
        self.upper_hsv[0] = (h + self.h_tol//2 + 180) % 180

        self.lower_hsv[1] = max(0, s - self.s_tol//2)
        self.upper_hsv[1] = min(255, s + self.s_tol//2)

        self.lower_hsv[2] = max(0, v - self.v_tol//2)
        self.upper_hsv[2] = min(255, v + self.v_tol//2)

        # optionally add logs
        print(f"Lower: {self.lower_hsv}\nUpper: {self.upper_hsv}\n")
    
def sample_color(event, x, y, flags,
                 param: tuple[cv2.typing.MatLike, CustomColor]):
    """Mouse callback for sampling color at cursor position on click.
    
    Requires the hsv frame parameter and the custom color object.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # parameters:
        frame_hsv = param[0]
        custom_color = param[1]
        
        hsv = frame_hsv[y, x]
        # print(f"Captured: {hsv}")
        
        # color update
        custom_color.update(hsv)

