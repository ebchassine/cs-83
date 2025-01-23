import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.ndimage import sobel, convolve
import cv2
import os 

def myImageFilter(img0, h):
    return convolve(img0, h, mode='constant', cval=0.0)

def myEdgeFilter(img0, sigma):
    # Gaussian Filter w/myImageFilter
    hsize = 2 * int(np.ceil(3 * sigma)) + 1  # Size of the Gaussian filter
    gaussian_kernel = gaussian(hsize, std=sigma).reshape(-1, 1)
    gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
    smoothed_img = myImageFilter(img0, gaussian_kernel_2d)

    # print(gaussian_kernel_2d)
    # Compute Gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    imgx = myImageFilter(smoothed_img, sobel_x)  # Gradient in x-direction
    imgy = myImageFilter(smoothed_img, sobel_y)  # Gradient in y-direction

    # Gradient magnitude and direction
    magnitude = np.hypot(imgx, imgy)  
    direction = np.arctan2(imgy, imgx) * (180 / np.pi)
    direction[direction < 0] += 180  # Map directions to [0, 180]

    #  Non-Maximum Suppression
    angle = np.round(direction / 45) * 45  # Map to nearest 0, 45, 90, or 135 degrees

    # Pad the magnitude array
    padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant') #mode=edge?

    # Vectorized Non-Maximum Suppression
    neighbors = {
        0: (padded_magnitude[1:-1, 2:], padded_magnitude[1:-1, :-2]),
        45: (padded_magnitude[:-2, 2:], padded_magnitude[2:, :-2]),
        90: (padded_magnitude[2:, 1:-1], padded_magnitude[:-2, 1:-1]),
        135: (padded_magnitude[2:, 2:], padded_magnitude[:-2, :-2])
    }

    suppressed = np.zeros_like(magnitude)
    for a in [0, 45, 90, 135]:
        q, r = neighbors[a]
        mask = angle == a
        suppressed[mask] = magnitude[mask] * ((magnitude[mask] >= q[mask]) & (magnitude[mask] >= r[mask]))

    return suppressed
    # return None 

def nms()

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
