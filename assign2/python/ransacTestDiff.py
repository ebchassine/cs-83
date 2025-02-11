import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.feature
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches

def compare_images(image1, image2):
    # Match features between images
    matches, locs1, locs2 = matchPics(image1, image2)
    
    if len(matches) == 0:
        print("No matches found.")
        return None, None
    
    x1 = locs1[matches[:, 0]][::-1]
    x2 = locs2[matches[:, 1]][::-1]
    
    # Compute homography using RANSAC
    H, inliers = computeH_ransac(x1, x2)
    
    # Plot matches
    plotMatches(image1, image2, matches, locs1, locs2)
    
    print(f"Number of inliers: {np.sum(inliers)}")
    return H, inliers

# Load images
img1_path = "./data/cv_cover.jpg"
img2_path = "./data/cv_desk.png"
image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)

if image1 is None or image2 is None:
    raise ValueError("Error loading images. Check file paths.")

# Compare original images
H, inliers = compare_images(image1, image2)
