import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.ndimage import sobel, convolve
import cv2
import os 
from myImageFilter import myImageFilter

# def myImageFilter(img0, h):
#     return convolve(img0, h, mode='constant', cval=0.0)

def myEdgeFilter(img0, sigma):
    # Gaussian Filter w/myImageFilter
    hsize = 2 * (np.ceil(3 * sigma)) + 1  # Size of the Gaussian filter
    gaussian_kernel = gaussian(hsize, std=sigma).reshape(-1, 1)
    gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
    smoothed_img = myImageFilter(img0, gaussian_kernel_2d)

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

    #  Non-Maximum Suppression Part 
    angle = (np.round(direction / 45) * 45) % 180  # Map to nearest 0, 45, 90, or 135 degrees

    # padding the magnitude array
    padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    # init output array
    suppressed = np.zeros_like(magnitude)

    # Non max supression 
    for i in range(1, magnitude.shape[0] + 1): 
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

# def main():
    # Input image path
    # input_image_path = input("../data/img01.jpg")
    # img0 = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

#     if img0 is None:
#         print("Error: Unable to read the input image. Make sure the path is correct.")
#         return

#     # Input sigma value
#     sigma = float(15.0)

#     # Perform edge detection
#     img1 = myEdgeFilter(img0, sigma)

#     # Display and save results
#     results_dir = "../results"
#     # os.makedirs(results_dir, exist_ok=True)

#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(img0, cmap='gray')
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Edge Magnitude Image")
#     plt.imshow(img1, cmap='gray')
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

#     # Save the output
#     output_image_path = os.path.join(results_dir, "edge_magnitude.png")
#     cv2.imwrite(output_image_path, (img1 / img1.max() * 255).astype(np.uint8))
#     print(f"Edge magnitude image saved at {output_image_path}")

# if __name__ == "__main__":
#     main()
