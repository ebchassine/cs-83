import numpy as np
# import cv2  # For cv2.dilate function

# img_hough (np.ndarray): The Hough transform accumulator (2D array).
# nLines (int): The number of lines to 
def myHoughLines(img_hough, nLines):
    indices = np.argpartition(img_hough.flatten(), -nLines)[-nLines:] # Filters top nLines from the indices
    indices = indices[np.argsort(img_hough.flatten()[indices])[::-1]] # Sorts them in descending order

    rows, cols = img_hough.shape  
    rhos = indices // cols  # int division to get row indices
    thetas = indices % cols  # mod to get column indices

    return rhos, thetas
