import torch
import numpy as np
import time

# Parameters
num_functions = 2000  # Number of functions to generate
num_points = 51  # Number of points along one dimension for the grid

def generate_circle_points(num_points):
    """Generate evenly spaced points along the unit circle's boundary."""
    theta = torch.linspace(0, 2 * np.pi, num_points)
    x = torch.cos(theta)
    y = torch.sin(theta)
    circle_points = torch.stack((x, y), dim=1)
    return circle_points

def generate_circle_inner_points(num_points):
    """Generate grid points inside the unit circle."""
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Filter points outside the unit circle
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_circle = np.sum(grid_points ** 2, axis=1) <= 1
    grid_points = grid_points[inside_circle]
    circle_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return circle_inner_points

def generate_outside_circle_sources(num_sources):
    """Generate random point sources outside the unit circle."""
    sources = []
    for _ in range(num_sources):
        r = torch.randn(1).abs().item() + 1.001  # Radius greater than 1 to ensure points are outside
        theta = torch.rand(1).item() * 2 * np.pi  # Angle in [0, 2*pi]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        weight = torch.randn(1)  # Random weight for the source
        sources.append((x, y, weight))
    return sources

def calculate_laplace_solution(points, sources):
    """Calculate Laplace equation solution as a linear combination of basis functions."""
    results = []
    for cx, cy, weight in sources:
        x_minus_cx = points[:, 0] - cx
        y_minus_cy = points[:, 1] - cy
        function_value = weight * 0.5 * torch.log(x_minus_cx ** 2 + y_minus_cy ** 2)
        results.append(function_value)

    # Sum the contributions from all sources using their weights
    weighted_sum = torch.stack(results, dim=1).sum(dim=1, keepdim=True)
    return weighted_sum

def generate_training_data(num_networks, num_points):
    """Generate training data combining boundary and inner points."""
    all_data = []

    # Generate fixed grid points inside the unit circle
    points = generate_circle_inner_points(num_points)
    # Generate points along the boundary of the unit circle
    boundary = generate_circle_points(100)

    for _ in range(num_networks):
        # Generate random point sources
        sources = generate_outside_circle_sources(10)  # Assume 10 sources

        # Compute solution at inner points
        u_val = calculate_laplace_solution(points, sources).view(1, -1)
        # Compute solution at boundary points
        u_b = calculate_laplace_solution(boundary, sources).view(1, -1)

        # Combine boundary and inner solutions
        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value  # Normalize to avoid large value ranges
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """Save the training data to a text file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Execution starts here
start_time = time.time()
file_path = f"data/MADlaplace2D2circ_{num_functions, num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
