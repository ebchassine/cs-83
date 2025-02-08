import numpy as np
import cv2
from matchPics import matchPics


def computeH(x1, x2):
	#Compute the homography between two sets of points
	assert x1.shape == x2.shape, "Unequal shapes between x1, x2"
	N = x1.shape[0]

    # Construct the A matrix
	A = []
	for i in range(N):
		x_1, y_1 = x1[i]
		x_2, y_2 = x2[i]
		# Add to A two rows at a time 
		A.append([-x_2, -y_2, -1, 0, 0, 0, x_1*x_2, x_1*y_2, x_1])
		A.append([0, 0, 0, -x_2, -y_2, -1, y_1*x_2, y_1*y_2, y_1])

	A = np.array(A) 
	# print(A)
	# do SVD of A 
	U, S, Vt = np.linalg.svd(A)

    # The solution to Ah = 0 is the last column of V (corresponding to the 	 singular value)
	H = Vt[-1].reshape(3, 3)
	# print(np.min(Vt))
	# print("Computed Homography Matrix:\n", H)
	H /= H[-1, -1]
	return H

# x1 = np.array([[100, 200], [200, 300], [300, 400], [400, 500]])
# x2 = np.array([[50, 150], [150, 250], [250, 350], [350, 450]])

# H = computeH(x1, x2)
# print("Homography Matrix:\n", H)


def computeH_norm(x1, x2):
	#Q3.7
	def normalize(coords): # input: homogenous coords, output: normalized coords, T 3x3 matrix
		#Compute the centroid of the points
		centroid = np.mean(coords, axis=0)
		#Shift the origin of the points to the centroid
		shifted = coords - centroid 
		#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
		normalized = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
		scale = np.sqrt(2) / normalized if normalized > 0 else 1
		T = np.array([
			[scale, 0, -scale * centroid[0]],
			[0, scale, -scale * centroid[1]],
			[0, 0, 1]
		])

		normalized_points = np.column_stack((coords, np.ones(coords.shape[0])))
		normalized_points = (T @ normalized_points.T).T[:, :2]

		return normalized_points, T
		
	#Similarity transform 1, transform 2
	x1_norm, T1 = normalize(x1)
	x2_norm, T2 = normalize(x2)

	# DEBUGGING 
	print("Normalized x1:\n", x1_norm)
	print("Normalized x2:\n", x2_norm)

	# Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	# Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	# H2to1 /= H2to1[2, 2]  # Normalize 
	
	return H2to1

def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	max_inliers = 0
	bestH2to1 = None
	inliers = None
	num_iters=10000
	tol=2

	N = x1.shape[0]
	for _ in range(num_iters):
		# Randomly sample 4 points
		indices = np.random.choice(N, 4, replace=False)
		H = computeH(x1[indices], x2[indices])

		# x2 points
		x2_homog = np.column_stack((x2, np.ones(N)))
		x2_transformed = (H @ x2_homog.T).T
		x2_transformed /= x2_transformed[:, 2].reshape(-1, 1)  # Convert to Cartesian

		# Euclidean distance
		distances = np.linalg.norm(x1 - x2_transformed[:, :2], axis=1)
		# x1_homog = np.column_stack((x1, np.ones(x1.shape[0])))
		# errors = np.sum((x1_homog - x2_transformed) ** 2, axis=1)  # Compute squared homogeneous error
		# inliers_mask = errors < tol**2  # Squared tolerance threshold
		
		# count inliers
		inliers_mask = distances < tol
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


def test_planarH():
    # Load images
    cv_cover = cv2.imread("./data/cv_cover.jpg", cv2.IMREAD_COLOR)
    cv_desk = cv2.imread("./data/cv_desk.png", cv2.IMREAD_COLOR)
    
    if cv_cover is None or cv_desk is None:
        print("Error: One or more images could not be loaded. Check file paths.")
        return

    # Get corresponding points using matchPics
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
    x1 = locs1[matches[:, 0], :]
    x2 = locs2[matches[:, 1], :]
    
    print("Testing computeH...")
    H = computeH(x1, x2)
    print("Homography Matrix (H):\n", H)

    print("\nTesting computeH_norm...")
    H_norm = computeH_norm(x1, x2)
    print("Normalized Homography Matrix (H_norm):\n", H_norm)

    print("\nTesting computeH_ransac...")
    bestH, inliers = computeH_ransac(x1, x2)
    print("Homography Matrix from RANSAC:\n", bestH)
    print("Inliers Mask:\n", inliers)

    print("\nTesting compositeH...")

    img = cv2.imread("./data/cv_desk.png")  
    template = cv2.imread("./data/hp_cover.jpg")

    if img is not None and template is not None:
        composite = compositeH(bestH, template, img)
        cv2.imwrite("../results/composite_test.jpg", composite)
        cv2.imshow("Composite Image", composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid Images for test")

if __name__ == "__main__":
    test_planarH()
