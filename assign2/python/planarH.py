import numpy as np
import cv2
from matchPics import matchPics
# import random # used for debugging 

def computeH(x1, x2):
	# assert x1.shape == x2.shape, "Unequal shapes between x1, x2"
	N = x1.shape[0]

	A = []
	for i in range(0, N):
		x_1, y_1 = x1[i] # x' y'
		x_2, y_2 = x2[i] # x y
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
		centroid = np.mean(coords, axis=0)
		shifted = coords - centroid 

		normalized = np.max(np.sqrt(np.sum(shifted**2, axis=1)))
		scale = np.sqrt(2) / normalized if normalized > 0 else 1
		T = np.array([
			[scale, 0, -scale * centroid[0]],
			[0, scale, -scale * centroid[1]],
			[0, 0, 1]
		])

		normalized_points = np.column_stack((coords, np.ones(coords.shape[0])))
		normalized_points = (T @ normalized_points.T).T[:, :2]

		return normalized_points, T
		
	x1_norm, T1 = normalize(x1)
	x2_norm, T2 = normalize(x2)

	# print("Normalized x1:\n", x1_norm)
	# print("Normalized x2:\n", x2_norm)
	H_norm = computeH(x1_norm, x2_norm)
	H2to1 = H_norm @ T2
	H2to1 = np.linalg.inv(T1) @ H2to1
	
	return H2to1

def computeH_ransac(x1, x2):
	#Q3.8
	max_inliers = 0
	bestH2to1 = None
	inliers = None
	iters=100
	threshold=1

	N = x1.shape[0]
	for _ in range(iters):
		indices = np.random.choice(N, 4, replace=False) #replace parameter gets rid of duplicate selections 
		# print("X1 SHAPE", x1[indices].shape)
		H = computeH_norm(x1[indices], x2[indices])
		
		x2_homog = np.column_stack((x2, np.ones(N)))
		x2_transformed = (H @ x2_homog.T).T
		x2_transformed /= x2_transformed[:, 2].reshape(-1, 1)  # convert back from homog 
		distances = np.linalg.norm(x1 - x2_transformed[:, :2], axis=1)

		inliers_mask = distances < threshold
		num_inliers = np.sum(inliers_mask)

		if num_inliers > max_inliers:
			max_inliers = num_inliers
			bestH2to1 = H
			inliers = inliers_mask

	return bestH2to1, inliers

def compositeH(H2to1, template, img):
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H2to1)

    warped_template = cv2.warpPerspective(template, H_inv, (w, h))
	
    composite_img = img.copy()
    composite_img[warped_template > 0] = warped_template[warped_template > 0]

    return composite_img