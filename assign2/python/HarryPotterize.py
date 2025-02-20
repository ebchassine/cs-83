import numpy as np
import cv2
import skimage.io 
import skimage.color
import skimage.transform
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

cv_cover = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2RGB) # for some reason when I was testing before, some of the images were gray
cv_desk = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2RGB)
hp_cover = cv2.cvtColor(hp_cover, cv2.COLOR_BGR2RGB)

reshaped_hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
hp_cover = reshaped_hp_cover

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

pts1 = locs1[matches[:, 0]]  # Points in cv_cover
pts2 = locs2[matches[:, 1]]  # Corresponding points in cv_desk

pts1[:, [0, 1]] = pts1[:, [1, 0]]
pts2[:, [0, 1]] = pts2[:, [1, 0]]


H2to1, inliers = computeH_ransac(pts1, pts2)

composite_img = compositeH(H2to1, hp_cover, cv_desk)

cv2.imwrite('../resultschat/harry_potterized.jpg', cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))
skimage.io.imshow(composite_img)
skimage.io.show()
