import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
import skimage.transform
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q3.9
def harry_potterize():
    cv_cover = cv2.imread('./data/cv_cover.jpg')
    cv_desk = cv2.imread('./data/cv_desk.png')
    hp_cover = cv2.imread('./data/hp_cover.jpg')

    # Check if images loaded correctly
    if cv_cover is None or cv_desk is None or hp_cover is None:
        print("Error: One or more images could not be loaded. Check file paths.")
        return

    # gray_cover = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
    # gray_desk = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2GRAY)

    # Match keypoints between cover and desk - note that the images might be converted to gray within matchPics
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)  # ✅ Correct: pass color images

    # Extract matching points
    x1 = locs1[matches[:, 0], :]
    x2 = locs2[matches[:, 1], :]

    # Compute homography using RANSAC
    H2to1, inliers = computeH_ransac(x1, x2)

    # Warp hp_cover image to match the book's perspective
    hp_warped = cv2.warpPerspective(hp_cover, H2to1, (cv_desk.shape[1], cv_desk.shape[0]))

    # Composite the images
    composite_img = compositeH(H2to1, hp_cover, cv_desk)

    # Save and display results
    cv2.imwrite('../results/composite_img.jpg', composite_img)
    skimage.io.imshow(composite_img)
    skimage.io.show()

if __name__ == "__main__":
    harry_potterize()