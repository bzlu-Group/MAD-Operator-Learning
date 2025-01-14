import numpy as np
from sklearn import gaussian_process as gp
from scipy.ndimage import gaussian_filter

def generate_gps(a, b, features, length_scalel, sample_train_num):
    """
    Generate samples from a Gaussian Process with RBF kernel.

    Args:
        a: Lower bound of the input domain.
        b: Upper bound of the input domain.
        features: Number of points in the input domain.
        length_scalel: Length scale parameter for the RBF kernel.
        sample_train_num: Number of samples to generate.

    Returns:
        gps: Generated samples, shape (features, sample_train_num).
    """
    x = np.linspace(a, b, num=features)[:, None]  # Input domain
    kernel_matrix = gp.kernels.RBF(length_scale=length_scalel)(x)  # RBF kernel
    L = np.linalg.cholesky(kernel_matrix + 1e-13 * np.eye(features))  # Cholesky decomposition
    gps = L @ np.random.randn(features, sample_train_num)  # Generate samples
    return gps

def generate_gps_circ(a, b, features, length_scalel, sample_train_num):
    """
    Generate circular samples by removing linear trends.

    Args:
        a: Lower bound of the input domain.
        b: Upper bound of the input domain.
        features: Number of points in the input domain.
        length_scalel: Length scale parameter for the RBF kernel.
        sample_train_num: Number of samples to generate.

    Returns:
        adjusted_gps: Circular samples, shape (sample_train_num, features - 1).
    """
    gps = generate_gps(a, b, features, length_scalel, sample_train_num).T

    # Compute linear trends for each sample
    start_vals = gps[:, 0]  # Start values for each sample
    end_vals = gps[:, -1]  # End values for each sample
    linear_trend = np.linspace(start_vals[:, None], end_vals[:, None], num=features, axis=1).squeeze()

    # Adjust samples by removing linear trends
    adjusted_gps = gps - linear_trend
    
    # Return adjusted samples excluding the last column
    return adjusted_gps[:, :-1]

def generate_smooth_random_field(size, sigma, sample_train_num):
    """
    Generate smooth random fields using Gaussian smoothing.

    Args:
        size: Size of the random field (size x size grid).
        sigma: Standard deviation for the Gaussian smoothing.
        sample_train_num: Number of samples to generate.

    Returns:
        smooth_fields: Array of smooth random fields, shape (sample_train_num, size, size).
    """
    # Generate random fields and apply Gaussian smoothing
    random_fields = np.random.randn(sample_train_num, size, size)
    smooth_fields = np.array([gaussian_filter(field, sigma=sigma) for field in random_fields])
    return smooth_fields

def generate_smooth_random_field_any(points, sigma, sample_train_num):
    """
    Generate smooth random fields for arbitrary points.

    Args:
        points: Array of coordinates representing the region of interest.
        sigma: Standard deviation for Gaussian smoothing.
        sample_train_num: Number of samples to generate.

    Returns:
        smooth_fields: Smooth random fields at given points, shape (sample_train_num, points.shape[0]).
    """
    smooth_fields = []

    for _ in range(sample_train_num):
        # Generate random values at given points
        random_values = np.random.randn(points.shape[0])
        
        # Apply Gaussian smoothing to the random values
        smooth_random_values = gaussian_filter(random_values, sigma=sigma)
        
        smooth_fields.append(smooth_random_values)

    return np.array(smooth_fields)




    