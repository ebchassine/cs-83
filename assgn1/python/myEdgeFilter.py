import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
# from scipy.ndimage import convolve
import cv2
# import os 
from myImageFilter import myImageFilter

# def myImageFilter(img0, h):
#     padded_img = np.pad(img0, pad_width=1, constant_values=0)
#     return convolve(padded_img, h, mode='constant', cval=0)


def myEdgeFilter(img0, sigma):
    # Gaussian Filter w/myImageFilter
    hsize = 2 * (np.ceil(3 * sigma)) + 1  # Size of the Gaussian filter
    gaussian_kernel = gaussian(hsize, std=sigma).reshape(-1, 1)
    gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
    gaussian_kernel_normalize = gaussian_kernel_2d / np.sum(gaussian_kernel_2d) 
    smoothed_img = myImageFilter(img0, gaussian_kernel_normalize)

    # print(gaussian_kernel_2d)
    # Compute Gradients w/ Sobel Kernel
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])

    imgx = myImageFilter(smoothed_img, sobel_x)  # Gradient in x-direction
    imgy = myImageFilter(smoothed_img, sobel_y)  # Gradient in y-direction

    # Gradient magnitude and direction
    magnitude = np.hypot(imgx, imgy)  # function that replaces sqrt(x^2 + y^2)

    direction = np.arctan2(imgy, imgx) * (180 / np.pi)
    direction[direction < 0] += 180  # Map directions to [0, 180] for any negative values (seen by filter)

    angle = (np.round(direction / 45) * 45) % 180  # Map to nearest 0, 45, 90, or 135 degrees
    padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    # init output array
    suppressed = np.zeros_like(magnitude)
    # print(angle)
    # Non max supression 
    for i in range(1, magnitude.shape[0] + 1):  # Note nested loop DOES run in permitted runtime 
        for j in range(1, magnitude.shape[1] + 1):
            grad_dir = angle[i - 1, j - 1]
            if grad_dir == 0:  # Horizontal
                neighbors = [padded_magnitude[i, j + 1], padded_magnitude[i, j - 1]]
            elif grad_dir == 45:  # Diagonal 
                neighbors = [padded_magnitude[i - 1, j + 1], padded_magnitude[i + 1, j - 1]]
            elif grad_dir == 90:  # Vertical
                neighbors = [padded_magnitude[i + 1, j], padded_magnitude[i - 1, j]]
            elif grad_dir == 135:  # Diagonal
                neighbors = [padded_magnitude[i + 1, j + 1], padded_magnitude[i - 1, j - 1]]
            else:
                continue

            if magnitude[i - 1, j - 1] >= max(neighbors):
                suppressed[i - 1, j - 1] = magnitude[i - 1, j - 1]
            else:
                suppressed[i - 1, j - 1] = 0
    # return suppressed
    # Dilation and Erosion to Remove Noise
    kernel = np.ones((3, 3), np.uint8)  # Structuring kernel (the larger the kernel, the larger of area of consideration for dilation), I found 2x2 was sweetspot 
    dilated = cv2.dilate(suppressed, kernel, iterations=1) # as per the instructions 
    cleaned = cv2.erode(dilated, kernel, iterations=1) # I found that this thinned the lines back to a similar thickness that was done by the dilation
    # Apply Thresholding (Optional, to further reduce noise)
    threshold = 0.05 * cleaned.max()
    cleaned[cleaned < threshold] = 0

    return cleaned

""" OH Version - I tried to implement the NMS from office hours we were unable to resolve it -- 
I went back to office hours on Sunday and TA Atul said the orignal aproach with nested loop was fine as long as it 
ran within permitted runtime (which it does)

def nms(magnitude, angle):

    angle = angle % 180

    kernels = {
        90: np.array([[0, 1, 0], 
                     [0, 1, 0], 
                     [0, 1, 0]], np.uint8),
        45: np.array([[0, 0, 1], 
                      [0, 1, 0], 
                      [1, 0, 0]], np.uint8),
        0: np.array([[0, 0, 0], 
                      [1, 1, 1], 
                      [0, 0, 0]], np.uint8),
        135: np.array([[1, 0, 0], 
                       [0, 1, 0], 
                       [0, 0, 1]], np.uint8)
    }
     
    suppressed = np.copy(magnitude)
    res = np.zeros_like(magnitude, dtype=np.float32) # init output 
    for direction, kernel in kernels.items():
        # convolved = myImageFilter(magnitude, kernel)
        convolved = cv2.dilate(magnitude, kernel, iterations=1)
        valid = (angle > direction - 22.5) & (angle <= direction + 22.5)
# 157-180
        suppressed = np.where((valid & (magnitude >= convolved)), magnitude, 0)
        res = res + suppressed

    return res
"""

""" Test script from TA 
import time
# Record start time
start_time = time.time()

img0 = cv2.imread("../data/img01.jpg")
img0 = cv2.cvtColor(src=img0, code=cv2.COLOR_BGR2GRAY)
img0 = img0.astype(np.float64)
# cv2.imshow("display image", img0)
# cv2.waitKey(0)
nms_image = myEdgeFilter(img0, 3)
# cv2.imshow("NMS Image", nms_image)
# cv2.waitKey(0)
cv2.imwrite('../results/img01_cleaned.jpg', nms_image)

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time} seconds")

"""