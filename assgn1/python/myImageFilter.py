import numpy as np
from numpy.lib.stride_tricks import as_strided

def myImageFilter(img0, h):
    # assert h.shape[0] % 2 == 1 and h.shape[1] % 2 == 1, "Kernel dimensions must be odd."

    # Get dimensions of the image and kernel
    img_height, img_width = img0.shape
    kernel_height, kernel_width = h.shape

    # Pad the input image to handle edges
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_img = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Create sliding window view of the image using as_strided
    output_shape = (img_height, img_width, kernel_height, kernel_width)
    strides = (
        padded_img.strides[0],  # Stride for rows of the image
        padded_img.strides[1],  # Stride for columns of the image
        padded_img.strides[0],  # Kernel row stride
        padded_img.strides[1],  # Kernel column stride
    )
    sliding_windows = as_strided(padded_img, shape=output_shape, strides=strides)

    # Perform convolution by multiplying kernel with sliding windows and summing
    img1 = np.einsum('ijkl,kl->ij', sliding_windows, h)

    return img1


if __name__ == "__main__":
    # Example grayscale image
    img = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=np.float32)

    # Example kernel (3x3 Gaussian-like filter)
    kernel = np.array([[0, 0, 0],
                       [0, 2, 0],
                       [0, 0, 0]], dtype=np.float32) / 16.0

    # Apply the filter
    filtered_img = myImageFilter(img, kernel)

    print("Original Image:\n", img)
    print("Filtered Image:\n", filtered_img)