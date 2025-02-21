"""
Homework 5
Submission Functions
"""

# import packages here

import helper
import numpy as np
import scipy.optimize
import numpy.linalg as la

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""

# def eight_point(pts1, pts2, M):
#     # replace pass by your implementation
#     # pass
#     T = np.array([[1/M, 0, -0.5], [0, 1/M, -0.5], [0, 0, 1]])
#     pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
#     pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
#     pts1_norm = (T @ pts1_h.T).T
#     pts2_norm = (T @ pts2_h.T).T

#     A = np.zeros((pts1.shape[0], 9))
#     for i in range(pts1.shape[0]):
#         x1, y1 = pts1_norm[i, :2]
#         x2, y2 = pts2_norm[i, :2]
#         A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

#     # Compute F using SVD
#     _, _, V = la.svd(A)
#     F = V[-1].reshape(3, 3)

#     # Enforce rank 2 constraint
#     U, S, V = la.svd(F)
#     S[-1] = 0
#     F = U @ np.diag(S) @ V

#     # Unnormalize F
#     F = T.T @ F @ T

#     return F

def eight_point(pts1, pts2, M):
    # Compute normalization matrices
    mean1 = np.mean(pts1, axis=0)
    mean2 = np.mean(pts2, axis=0)
    
    std1 = np.sqrt(2) / np.mean(np.linalg.norm(pts1 - mean1, axis=1))
    std2 = np.sqrt(2) / np.mean(np.linalg.norm(pts2 - mean2, axis=1))
    
    T1 = np.array([[std1, 0, -std1 * mean1[0]],
                    [0, std1, -std1 * mean1[1]],
                    [0, 0, 1]])
    
    T2 = np.array([[std2, 0, -std2 * mean2[0]],
                    [0, std2, -std2 * mean2[1]],
                    [0, 0, 1]])
    
    # Normalize points
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts1_norm = (T1 @ pts1_h.T).T
    pts2_norm = (T2 @ pts2_h.T).T
    
    # Construct matrix A
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1 = pts1_norm[i, :2]
        x2, y2 = pts2_norm[i, :2]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    # Compute F using SVD
    _, _, V = la.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint
    U, S, V = la.svd(F)
    S[-1] = 0  # Force rank 2
    F = U @ np.diag(S) @ V
    
    # Refine F using provided helper function
    F = helper.refineF(F, pts1_norm[:, :2], pts2_norm[:, :2])
    
    # Unnormalize F
    F = T2.T @ F @ T1
    
    return F



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    pass


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    pass


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
