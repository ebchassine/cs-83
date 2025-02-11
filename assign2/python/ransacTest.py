import numpy as np
import cv2
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import computeH_ransac

# Convert feature locations to OpenCV KeyPoint objects
def convert_to_keypoints(locs):
    return [cv2.KeyPoint(float(x), float(y), 1) for x, y in locs]

# Convert matches to OpenCV DMatch objects
def convert_to_dmatches(matches):
    return [cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _distance=0) for m in matches]

# Load the image as grayscale
image_path = "../data/cv_cover.jpg"
image = cv2.imread(image_path)

# Ensure the image was loaded correctly
if image is None:
    raise ValueError(f"Error loading image from {image_path}")

# Get image dimensions
rows, cols = image.shape[:2]

# Test RANSAC with the same image twice
matches, locs1, locs2 = matchPics(image, image)
x1 = locs1[matches[:, 0]]
x2 = locs2[matches[:, 1]]
H_same, inliers_same = computeH_ransac(x1, x2)

# Rotate the image by 10 degrees
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Test RANSAC with the rotated image
matches_rot, locs1_rot, locs2_rot = matchPics(image, rotated_image)
x1_rot = locs1_rot[matches_rot[:, 0]]
x2_rot = locs2_rot[matches_rot[:, 1]]
H_rot, inliers_rot = computeH_ransac(x1_rot, x2_rot)

# Display results
print(f"Number of inliers (same image): {np.sum(inliers_same)}")
print(f"Number of inliers (rotated image): {np.sum(inliers_rot)}")

# Convert feature locations to KeyPoint objects for visualization
keypoints1 = convert_to_keypoints(locs1)
keypoints2 = convert_to_keypoints(locs2)
keypoints1_rot = convert_to_keypoints(locs1_rot)
keypoints2_rot = convert_to_keypoints(locs2_rot)

# Convert matches to DMatch objects for visualization
matches_cv = convert_to_dmatches(matches)
matches_rot_cv = convert_to_dmatches(matches_rot)

# Draw matches
match_img_same = cv2.drawMatches(image, keypoints1, image, keypoints2, matches_cv, None)
match_img_rot = cv2.drawMatches(image, keypoints1_rot, rotated_image, keypoints2_rot, matches_rot_cv, None)

# Display the matches
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image Matches")
plt.imshow(match_img_same, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Rotated Image Matches")
plt.imshow(match_img_rot, cmap="gray")

plt.show()