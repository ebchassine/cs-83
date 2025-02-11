import cv2
import numpy as np
from matchPics import matchPics
from planarH import computeH, computeH_norm, computeH_ransac, compositeH

def main():
    # Load images
    img = cv2.imread("../data/cv_desk.png")
    template = cv2.imread("../data/cv_cover.jpg")

    matches, locs1, locs2 = matchPics(template, img)

    x1 = locs1[matches[:, 0]]#[::-1]
    x2 = locs2[matches[:, 1]]#[::-1]

    x1[:, [0,1]] = x1[:, [1,0]]
    x2[:, [0,1]] = x2[:, [1,0]] 
    # x1 = locs1[matches[:, 0]][::-1]
    # x2 = locs2[matches[:, 1]][::-1]

    print("\nTesting computeH...")
    H = computeH(x1, x2)
    print("Homography Matrix (H):\n", H)

    print("\nTesting computeH_norm...")
    H_norm = computeH_norm(x1, x2)
    print("Normalized Homography Matrix (H_norm):\n", H_norm)

    print("\nTesting computeH_ransac...")
    bestH, inliers = computeH_ransac(x1, x2)
    print(f"RANSAC Inliers: {np.sum(inliers)}/{len(inliers)}")
    print("Best Homography Matrix from RANSAC:\n", bestH)

    print("\nTesting compositeH...")

    composite = compositeH(bestH, template, img)
    cv2.imwrite("./results/composite_test.jpg", composite)
    cv2.imshow("Composite Image", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()