import numpy as np
from numpy.lib.stride_tricks import as_strided

def myImageFilter(img0, h):
    img_height, img_width = img0.shape
    kernel_height, kernel_width = h.shape

    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    # padded_img = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    padded_img = np.pad(img0, pad_width=pad_width, constant_values=0)

    # Create sliding window view of the image using as_strided - vectorized method of applying convolution
    # sources
    # https://jessicastringham.net/2018/01/01/einsum/
    # https://sangillee.com/2024-07-27-how-to-compute-faster-conv2d-python/
    # There were some warnings in the np documentation, but I didn't run into any obvious issues 
    output_shape = (img_height, img_width, kernel_height, kernel_width)
    strides = (
        padded_img.strides[0],  # Stride for rows of the image
        padded_img.strides[1],  # Stride for columns of the image
        padded_img.strides[0],  # Kernel row stride
        padded_img.strides[1],  # Kernel column stride
    )
    sliding_windows = as_strided(padded_img, shape=output_shape, strides=strides)
    # multiply kernel with sliding windows and summing, refer to blog sources 
    img1 = np.einsum('ijkl,kl->ij', sliding_windows, h)

    return img1