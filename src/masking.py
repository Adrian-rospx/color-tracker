"""Custom mask creation functions for abstraction"""

import cv2
import numpy as np

def createColorThreshold(img_hsv: cv2.typing.MatLike, 
                         lower: tuple[int], 
                         upper: tuple[int]) -> cv2.typing.MatLike:
    """For creating the color threshold mask used during the processing
    
    Handles red hue values, where the the upper value is lower than the first! 
    """
    if upper[0] < lower[0]:
        # when the hue value loops around the 180 degrees mark (red hue)
        mask_high_hue = cv2.inRange(img_hsv, lower, (180, upper[1], upper[2]))
        mask_low_hue = cv2.inRange(img_hsv, (0, lower[1], lower[2]), upper)
        
        # combine the two masks
        return cv2.bitwise_or(mask_low_hue, mask_high_hue)
    else:
        return cv2.inRange(img_hsv, lower, upper)

def applyCLAHE(img_hsv: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """CLAHE algorithm applied over hsv images"""
    h, s, v = cv2.split(img_hsv)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    v2 = clahe.apply(v)

    return cv2.merge((h, s, v2))

# Main masking function
def createMask(img: cv2.typing.MatLike, 
               lower_hsv: tuple[int], 
               upper_hsv: tuple[int]) -> cv2.typing.MatLike:
    """Current custom masking function. Applies the following:
    
    - HSV conversion
    - Color threshold mask creation
    - Gaussian blurring
    - Morphology denoising
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # apply CLAHE filter to hsv
    img_hsv = applyCLAHE(img_hsv)

    # color threshold mask creation
    mask = createColorThreshold(img_hsv, lower_hsv, upper_hsv)

    # blurring
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # morphology for denoising
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 2)

    return mask