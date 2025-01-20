import torch
import numpy as np
import time

# Parameters
num_functions = 2000
num_points = 51
min_distance = 0.001  # Minimum distance to avoid numerical instability

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

def generate_outside_sources(num_sources):
    """Generate random point sources located outside the computational domain."""
    sources = []

    for _ in range(num_sources):
        if torch.rand(1) > 0.5:
            if torch.rand(1).item() > 0.5:
                # Case 1: x outside [0, 1], y within [0, 1]
                x = torch.randn(1).item()
                y = torch.rand(1).item()
                x = x - min_distance if x <= 0 else x + 1 + min_distance
            else:
                # Case 2: y outside [0, 1], x within [0, 1]
                y = torch.randn(1).item()
                x = torch.rand(1).item()
                y = y - min_distance if y <= 0 else y + 1 + min_distance
        else:
            # Case 3: Both x and y outside [0, 1]
            x = torch.randn(1).item()
            y = torch.randn(1).item()
            x = x - min_distance if x <= 0 else x + 1 + min_distance
            y = y - min_distance if y <= 0 else y + 1 + min_distance
        
        weight = torch.randn(1)  # Random weight for the source
        sources.append((x, y, weight))

    return sources

def calculate_laplace_solution(points, sources):
    """Compute the Laplace solution at given points based on multiple sources."""
    results = []
    for cx, cy, weight in sources:
        x_minus_cx = points[:, 0] - cx
        y_minus_cy = points[:, 1] - cy
        function_value = weight * 0.5 * torch.log(x_minus_cx ** 2 + y_minus_cy ** 2)
        results.append(function_value)

    # Combine the contributions from all sources
    weighted_sum = torch.stack(results, dim=1).sum(dim=1, keepdim=True)
    return weighted_sum

def generate_training_data(num_networks, num_points):
    """Generate training data for Laplace solutions."""
    all_data = []

    h = 1 / (num_points - 1)
    x = np.arange(0, 1 + h, h)
    y = np.arange(0, 1 + h, h)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    points = torch.tensor(grid_points, dtype=torch.float32)

    for _ in range(num_networks):
        sources = generate_outside_sources(10)
        u_val = calculate_laplace_solution(points, sources).view(1, -1)
        u_b = calculate_laplace_solution(boundary, sources).view(1, -1)
        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """Save the training data to a file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/MADlaplace2D2_{num_functions, num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
