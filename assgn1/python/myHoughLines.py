import numpy as np
import cv2  # For cv2.dilate function

# img_hough (np.ndarray): The Hough transform accumulator (2D array).
# nLines (int): The number of lines to detect.
def myHoughLines(img_hough, nLines):
    # Normalize the accumulator
    normalized_hough = img_hough / np.max(img_hough)

    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(normalized_hough, kernel)  # Dilate the accumulator to find local maxima
    non_max_suppressed = np.where(normalized_hough == local_max, normalized_hough, 0)

    indices = np.argpartition(non_max_suppressed.flatten(), -nLines)[-nLines:]
    indices = indices[np.argsort(non_max_suppressed.flatten()[indices])[::-1]]

    rhos, thetas = np.unravel_index(indices, img_hough.shape)

    # Convert the indices into ρ and θ
    return rhos, thetas
