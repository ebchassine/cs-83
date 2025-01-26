import numpy as np

# Im (numpy.ndarray) - Input binary edge-detected image.
# rhoRes (float) - Resolution of the rho parameter.
# thetaRes (float) -  Resolution of the theta parameter in radians.
def myHoughTransform(Im, rhoRes, thetaRes):
    rows, cols = Im.shape
    max_dist = int(np.sqrt(rows**2 + cols**2))
    rhoScale = np.arange(0, max_dist + 1, rhoRes) # Range of [0, M] 
    # rhoScale = np.arange(0, max_dist, rhoRes)   
    thetaScale = np.arange(0, 2*np.pi, thetaRes)  # Range of [0, 2pi]

    hough_accumulator = np.zeros((len(rhoScale), len(thetaScale)), dtype=np.int64)
    # print(hough_accumulator.shape) 

    # Populate the Hough accumulator
    edge_points = np.argwhere(Im > 0)  # Find edge points in the image
    for y, x in edge_points:
        for theta_idx, theta in enumerate(thetaScale):
            rho = x * np.cos(theta) + y * np.sin(theta)
            # rho_idx = int((rho + max_dist) / rhoRes)
            if rho < 0: # skip neg vals 
                continue
            rho_idx =  np.argmin(abs(rhoScale - rho)) # scale index using argmin
            # print(rho_idx, theta_idx)
            # print(rho)
            hough_accumulator[rho_idx, theta_idx] += 1
    

    return hough_accumulator, rhoScale, thetaScale