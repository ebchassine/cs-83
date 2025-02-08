import numpy as np
import cv2
from matchPics import matchPics


def computeH(x1, x2):
	#Compute the homography between two sets of points
	assert x1.shape == x2.shape, "Unequal shapes between x1, x2"
	N = x1.shape[0]

    # Construct the A matrix
	A = []
	for i in range(0, N):
		x_1, y_1 = x1[i] # x' y'
		x_2, y_2 = x2[i] # x y
		# Add to A two rows at a time 
			#		[-x -y, -1, 0, 0, 0, xx', yx', x']
			#		[0, 0, 0, -x. -y -1. y'x, yy', y']
		A.append([-x_2, -y_2, -1, 0, 0, 0, x_1*x_2, x_1*y_2, x_1])
		A.append([0, 0, 0, -x_2, -y_2, -1, y_1*x_2, y_1*y_2, y_1])
	A = np.array(A) 

	U, S, V = np.linalg.svd(A)
 
	# H = np.reshape(V[-1], (3,3))
	H = V[-1].reshape(3,3)

	return H

def computeH_norm(x1, x2):
	#Q3.7
	def normalize(coords): # input: homogenous coords, output: normalized coords, T 3x3 matrix
		#Compute the centroid of the points
		centroid = np.mean(coords, axis=0)
		#Shift origin of the points to the centroid
		shifted = coords - centroid 
		#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
		normalized = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
		scale = np.sqrt(2) / normalized if normalized > 0 else 1
		T = np.array([
			[-scale, 0, scale * centroid[0]],
			[0, -scale, scale * centroid[1]],
			[0, 0, 1]
		])

		normalized_points = np.column_stack((coords, np.ones(coords.shape[0])))
		normalized_points = (T @ normalized_points.T).T[:, :2]

		return normalized_points, T
		
	x1_norm, T1 = normalize(x1)
	x2_norm, T2 = normalize(x2)

	# print("Normalized x1:\n", x1_norm)
	# print("Normalized x2:\n", x2_norm)
	# Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	
	return H2to1

def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	max_inliers = 0
	bestH2to1 = None
	inliers = None
	num_iters=20
	threshold=1

	N = x1.shape[0]
	for _ in range(num_iters):
		indices = np.random.choice(N, 4, replace=False) #replace parameter gets rid of duplicate selections 
		H = computeH(x1[indices], x2[indices])

		# x2 points
		x2_homog = np.column_stack((x2, np.ones(N)))
		x2_transformed = (H @ x2_homog.T).T
		x2_transformed /= x2_transformed[:, 2].reshape(-1, 1)  # Convert to Cartesian

		# Euclidean distance
		distances = np.linalg.norm(x1 - x2_transformed[:, :2], axis=1)
		# x1_homog = np.column_stack((x1, np.ones(x1.shape[0])))
		# errors = np.sum((x1_homog - x2_transformed) ** 2, axis=1)  # Compute squared homogeneous errormes
		
		# count inliers
		inliers_mask = distances < threshold
		num_inliers = np.sum(inliers_mask)

		if num_inliers > max_inliers:
			max_inliers = num_inliers
			bestH2to1 = H
			inliers = inliers_mask

	return bestH2to1, inliers


def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	h, w = img.shape[:2]

	# Warp the template using the inverse homography
	warped_template = cv2.warpPerspective(template, H2to1, (w, h))

	# Create a binary mask from the template
	# mask = (warped_template > 0).astype(np.uint8)
	mask = (warped_template[:, :, 0] > 0).astype(np.uint8)  # Convert to 1-channel mask
	mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand to 3-channel mask
	composite_img = cv2.addWeighted(img, 1, warped_template, 1, 0) * mask + img * (1 - mask)

	# Mask out the area in the original image
	img_masked = img * (1 - mask)

	# Combine the two images
	composite_img = img_masked + warped_template

	return composite_img

import random

# def main():
# 	# random.seed(1)
# 	# np.set_printoptions(suppress=True)

# 	x1 = np.array([[1, -3], [4, -4], [3, -5], [6, -6], [2, -7], [7, -8], [8, -9], [9, -10]])
# 	x2 = np.array([[100, -300], [400, -400], [300, -500], [600, -600], [200, -700], [700, -800], [800, -900], [900, -1000]])

# 	# print(computeH(x1, x2))

# 	# print(" ")
# 	# print(" ")
# 	# print(" ")

# 	print(computeH_norm(x1, x2))


# def main():
#     img = cv2.imread("./data/cv_desk.png", cv2.IMREAD_COLOR)
#     template = cv2.imread("./data/cv_cover.jpg", cv2.IMREAD_COLOR)

#     matches, locs1, locs2 = matchPics(template, img)
#     x1 = locs1[matches[:, 0], :]
#     x2 = locs2[matches[:, 1], :]
    
#     print("Testing computeH...")
#     H = computeH(x1, x2)
#     print("Homography Matrix (H):\n", H)

#     print("\nTesting computeH_norm...")
#     H_norm = computeH_norm(x1, x2)
#     print("Normalized Homography Matrix (H_norm):\n", H_norm)

#     print("\nTesting computeH_ransac...")
#     bestH, inliers = computeH_ransac(x1, x2)
#     print("Best Homography Matrix from RANSAC:\n", bestH)
#     print("Inliers Mask:\n", inliers)

#     print("\nTesting compositeH...")
#     if img is not None and template is not None:
#         composite = compositeH(bestH, template, img)
#         cv2.imwrite("./results/composite_test.jpg", composite)
#         cv2.imshow("Composite Image", composite)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("Invalid image files for compositeH test.")

# if __name__ == "__main__":
#     main()
