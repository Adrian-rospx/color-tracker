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

# Main masking function
def createMask(img: cv2.typing.MatLike, 
               lower_hsv: tuple[int], 
               upper_hsv: tuple[int]) -> cv2.typing.MatLike:
    """Current custom masking function. Applies the following:
    
    - hsv conversion
    - color threshold mask creation
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color threshold mask creation
    mask = createColorThreshold(img_hsv, lower_hsv, upper_hsv)

    # morphology for denoising
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask