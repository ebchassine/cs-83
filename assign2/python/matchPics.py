import numpy as np
import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

SIGMA = 0.15
def matchPics(I1, I2):
	#Convert Images to GrayScale
	I1_gray, I2_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	locs_i1, locs_i2 = corner_detection(I1_gray, SIGMA), corner_detection(I2_gray, SIGMA)
	
	desc1, locs1 = computeBrief(I1_gray, locs_i1)
	desc2, locs2 = computeBrief(I2_gray, locs_i2)

	matches = briefMatch(desc1, desc2, ratio=0.65)

	# plotMatches(I1, I2, matches, locs1, locs2)
	return matches, locs1, locs2

if __name__ == "__main__":
    I1 = cv2.imread("../data/cv_cover.jpg") 
    I2 = cv2.imread("../data/cv_desk.png")

    try:
        matches, locs1, locs2 = matchPics(I1, I2)
        plotMatches(I1, I2, matches, locs1, locs2)
    except ValueError as e:
        print(e)