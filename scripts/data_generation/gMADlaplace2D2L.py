import torch
import numpy as np
import time

# Parameters
num_functions = 2000  # Number of functions to generate
num_points = 51  # Number of points for the grid
min_distance = 0.001  # Minimum distance from sources to the L-shaped region boundary

def generate_l_shape_boundary_points(target_spacing):
    """
    Generate boundary points of the L-shaped region with specified spacing between points.
    """
    length_1, length_2, length_3 = 1, 0.5, 0.5
    length_4, length_5, length_6 = 0.5, 0.5, 1

    # Number of points for each edge based on spacing
    num_points_1 = int(length_1 / target_spacing) + 1
    num_points_2 = int(length_2 / target_spacing) + 1
    num_points_3 = int(length_3 / target_spacing) + 1
    num_points_4 = int(length_4 / target_spacing) + 1
    num_points_5 = int(length_5 / target_spacing) + 1
    num_points_6 = int(length_6 / target_spacing) + 1

    # Define boundary points for each edge
    x1 = torch.linspace(0, 1, num_points_1)
    y1 = torch.zeros(num_points_1)

    x2 = torch.full((num_points_2,), 1)
    y2 = torch.linspace(0, 0.5, num_points_2)

    x3 = torch.linspace(1, 0.5, num_points_3)
    y3 = torch.full((num_points_3,), 0.5)

    x4 = torch.full((num_points_4,), 0.5)
    y4 = torch.linspace(0.5, 1, num_points_4)

    x5 = torch.linspace(0.5, 0, num_points_5)
    y5 = torch.ones(num_points_5)

    x6 = torch.zeros(num_points_6)
    y6 = torch.linspace(1, 0, num_points_6)

    # Combine all boundary points
    boundary_points = torch.cat([
        torch.stack((x1, y1), dim=1),
        torch.stack((x2, y2), dim=1),
        torch.stack((x3, y3), dim=1),
        torch.stack((x4, y4), dim=1),
        torch.stack((x5, y5), dim=1),
        torch.stack((x6, y6), dim=1)
    ], dim=0)

    # Remove duplicate points
    boundary_points = torch.unique(boundary_points, dim=0)
    return boundary_points

def generate_l_shape_inner_points(num_points):
    """
    Generate points inside the L-shaped region on a uniform grid.
    """
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter points inside the L-shaped region
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_l_shape = (grid_points[:, 0] <= 0.5) | (grid_points[:, 1] <= 0.5)
    grid_points = grid_points[inside_l_shape]

    return torch.tensor(grid_points, dtype=torch.float32)

def generate_outside_l_shape_sources(num_sources):
    """
    Generate random sources outside the L-shaped region.
    """
    sources = []
    for _ in range(num_sources):
        mode = np.random.choice(['outside_box', 'inside_top_right'])
        if mode == 'outside_box':
            # Generate sources outside the bounding box [0, 1]^2
            x, y = torch.randn(2).tolist()
            x += -min_distance if x <= 0 else 1 + min_distance
            y += -min_distance if y <= 0 else 1 + min_distance
        elif mode == 'inside_top_right':
            # Generate sources in the top-right corner [0.5, 1]^2
            x = torch.randn(1).abs().item() + 0.5 + min_distance
            y = torch.randn(1).abs().item() + 0.5 + min_distance

        weight = torch.randn(1)
        sources.append((x, y, weight))

    return sources

def calculate_laplace_solution(points, sources):
    """
    Calculate the solution of the Laplace equation for given sources at specific points.
    """
    results = []
    for cx, cy, weight in sources:
        x_minus_cx = points[:, 0] - cx
        y_minus_cy = points[:, 1] - cy
        function_value = weight * 0.5 * torch.log(x_minus_cx ** 2 + y_minus_cy ** 2)
        results.append(function_value)

    # Combine results using the weights
    return torch.stack(results, dim=1).sum(dim=1, keepdim=True)

def generate_training_data(num_networks, num_points):
    """
    Generate training data for the L-shaped region.
    """
    all_data = []
    points = generate_l_shape_inner_points(num_points)
    boundary = generate_l_shape_boundary_points(1 / (num_points - 1))

    for _ in range(num_networks):
        sources = generate_outside_l_shape_sources(10)  # Assume 10 random sources
        u_val = calculate_laplace_solution(points, sources).view(1, -1)
        u_b = calculate_laplace_solution(boundary, sources).view(1, -1)

        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """
    Save the training data to a file.
    """
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Main execution
start_time = time.time()
file_path = f"data/MADlaplace2D2L_{num_functions, num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
