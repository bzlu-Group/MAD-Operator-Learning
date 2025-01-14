import torch
import numpy as np
from scipy.special import j0, y0
import time
import math

# Parameters
num_functions = 2000
num_points = 51
k = 1  # Equation: △u+ku=0

min_distance = 0.001  # Minimum distance to avoid numerical instability

# Generate boundary points of a square
def generate_square_points(num_points):
    """Generate evenly spaced points along the boundary of a square."""
    points_per_edge = num_points
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge-1)), dim=1)
    edge2 = torch.stack((torch.ones(points_per_edge-1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge-1)), dim=1)
    edge4 = torch.stack((torch.zeros(points_per_edge-1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

boundary = generate_square_points(num_points)

# Generate point sources outside the domain
def generate_outside_sources(num_sources):
    """Generate random point sources located outside the computational domain."""
    sources = []

    for _ in range(num_sources):
        if torch.rand(1) > 0.5:
            if torch.rand(1).item() > 0.5:
                x = torch.randn(1).item()
                y = torch.rand(1).item()
                x = x - min_distance if x <= 0 else x + 1 + min_distance
            else:
                y = torch.randn(1).item()
                x = torch.rand(1).item()
                y = y - min_distance if y <= 0 else y + 1 + min_distance
        else:
            x = torch.randn(1).item()
            y = torch.randn(1).item()
            x = x - min_distance if x <= 0 else x + 1 + min_distance
            y = y - min_distance if y <= 0 else y + 1 + min_distance
        
        sources.append((x, y))

    return sources

# Generate training data
def generate_training_data(num_networks, num_points):
    """Generate training data using fundamental solutions."""
    all_data = []
    h = 1 / (num_points - 1)
    x = np.arange(0, 1 + h, h)
    y = np.arange(0, 1 + h, h)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    points = torch.tensor(grid_points, dtype=torch.float32)

    for _ in range(num_networks):
        sources = generate_outside_sources(10)
        j0_results, y0_results = [], []

        for (x_pole, y_pole) in sources:
            pole_location = np.array([x_pole, y_pole])
            r_values = np.linalg.norm(points.numpy() - pole_location, axis=1) * math.sqrt(k)
            j0_results.append(j0(r_values))
            y0_results.append(y0(r_values))

        c1 = np.random.randn(10)
        c2 = np.random.randn(10)
        u_val = sum(c1[i] * j0_results[i] + c2[i] * y0_results[i] for i in range(10))
        u_val = torch.tensor(u_val, dtype=torch.float32).view(1, -1)

        j0_boundary_results, y0_boundary_results = [], []

        for (x_pole, y_pole) in sources:
            pole_location = np.array([x_pole, y_pole])
            r_boundary = np.linalg.norm(boundary.numpy() - pole_location, axis=1) * math.sqrt(k)
            j0_boundary_results.append(j0(r_boundary))
            y0_boundary_results.append(y0(r_boundary))

        u_b = sum(c1[i] * j0_boundary_results[i] + c2[i] * y0_boundary_results[i] for i in range(10))
        u_b = torch.tensor(u_b, dtype=torch.float32).view(1, -1)

        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

# Save training data
def save_training_data(data, file_path):
    """Save training data to a file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/MAD{k}helmholtz_withoutsource_2D_{num_functions, num_points}.txt"

training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")


