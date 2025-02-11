from matchPics import matchPics 
import cv2
from helper import plotMatches

I1 = cv2.imread("../data/cv_cover.jpg") 
I2 = cv2.imread("../data/cv_desk.png")

try:
    matches, locs1, locs2 = matchPics(I1, I2)
    plotMatches(I1, I2, matches, locs1, locs2)
except ValueError as e:
    print(e)