"""Custom mask creation functions for abstraction"""

import cv2
import numpy as np

def preprocess(img_hsv: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Applies the following:
    
    - CLAHE
    - Gaussian blur
    """
    h, s, v = cv2.split(img_hsv)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    v2 = clahe.apply(v)

    # blurring the value channel
    v2 = cv2.GaussianBlur(v2, (7, 7), 0)

    return cv2.merge((h, s, v2))

def create_color_threshold(img_hsv: cv2.typing.MatLike, 
                         lower: tuple[int, int, int], 
                         upper: tuple[int, int, int]) -> cv2.typing.MatLike:
    """For creating the color threshold mask used during the processing
    
    Handles red hue values, where the the upper value is lower than the first!
    """
    if upper[0] < lower[0]:
        # when the hue value loops around the 180 degrees mark (red hue)
        mask_high_hue = cv2.inRange(img_hsv, lower, (179, upper[1], upper[2]))
        mask_low_hue = cv2.inRange(img_hsv, (0, lower[1], lower[2]), upper)
        
        # combine the two masks
        return cv2.bitwise_or(mask_low_hue, mask_high_hue)
    else:
        return cv2.inRange(img_hsv, lower, upper)

# Main masking function
def create_mask(img_hsv: cv2.typing.MatLike, 
               lower_hsv: tuple[int, int, int], 
               upper_hsv: tuple[int, int, int]) -> cv2.typing.MatLike:
    """Custom color masking function. Applies the following:
    
    - Color thresholding
    - Morphology denoising
    """
    # color threshold mask creation
    mask = create_color_threshold(img_hsv, lower_hsv, upper_hsv)

    # morphology for denoising
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 2)

    return mask