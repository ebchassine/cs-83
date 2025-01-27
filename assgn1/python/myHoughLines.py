import numpy as np
import cv2  # For cv2.dilate function

# img_hough (np.ndarray): The Hough transform accumulator (2D array).
# nLines (int): The number of lines to 
def myHoughLines(img_hough, nLines):
    normalized_hough = img_hough / np.max(img_hough)
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(normalized_hough, kernel)  # dilate the accumulator to find local maxima as per instrucitons 
    non_max_suppressed = np.where(normalized_hough == local_max, normalized_hough, 0)

    indices = np.argpartition(non_max_suppressed.flatten(), -nLines)[-nLines:] # Filters top nLines from the indices
    indices = indices[np.argsort(non_max_suppressed.flatten()[indices])[::-1]] # Sorts them in descending order

    rows, cols = img_hough.shape  
    rhos = indices // cols  # int division to get row indices
    thetas = indices % cols  # mod to get column indices

    return rhos, thetas
