import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    """
    Perform Hough Transform and detect lines using the Hough transform output with non-maximal suppression.

    Args:
        Im (numpy.ndarray): Input binary edge-detected image.
        rhoRes (float): Resolution of the rho parameter.
        thetaRes (float): Resolution of the theta parameter in radians.

    Returns:
        img_hough (numpy.ndarray): Hough transform accumulator.
        rhoScale (numpy.ndarray): Array of rho values corresponding to accumulator rows.
        thetaScale (numpy.ndarray): Array of theta values corresponding to accumulator columns.
    """
    # Compute the Hough accumulator
    rows, cols = Im.shape
    max_dist = int(np.sqrt(rows**2 + cols**2))
    rhoScale = np.arange(-max_dist, max_dist + 1, rhoRes)
    thetaScale = np.arange(0, np.pi, thetaRes)

    hough_accumulator = np.zeros((len(rhoScale), len(thetaScale)), dtype=np.int64)

    # Populate the Hough accumulator
    edge_points = np.argwhere(Im > 0)  # Find edge points in the image
    for y, x in edge_points:
        for theta_idx, theta in enumerate(thetaScale):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int((rho + max_dist) / rhoRes)
            hough_accumulator[rho_idx, theta_idx] += 1

    return hough_accumulator, rhoScale, thetaScale