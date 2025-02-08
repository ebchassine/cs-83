import numpy as np
import cv2
from matchPics import matchPics
import scipy.ndimage
import matplotlib.pyplot as plt

#Q3.5
#Read the image and convert to grayscale, if necessary
image_path = "./data/cv_cover.jpg"
img = cv2.imread(image_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

histo = [] 

for i in range(1,36): # Excluding 0 rotation - 350 degrees included in the graph - Kishmore advised us to
	theta = i * 10
	rotated_image =	 scipy.ndimage.rotate(img, theta) 
	matches, _, _ = matchPics(img, rotated_image)
	histo.append(len(matches))

# Display histogram
plt.figure(figsize=(8, 5))
plt.bar([angle for angle in range(0, 350, 10)], histo, width=10)
plt.title("BRIEF Descriptor Matching Across Rotations")
plt.xlabel("Rotation Angle (degrees)")
plt.ylabel("Number of Matches")

plt.show()
