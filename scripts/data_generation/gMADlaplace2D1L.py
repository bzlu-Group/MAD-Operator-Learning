import torch
import torch.nn as nn
import numpy as np
import time

# Parameters
num_functions = 2000  # Number of functions to generate
num_points = 51  # Number of points for internal grid and boundary

def generate_l_shape_boundary_points(target_spacing):
    """
    Generate boundary points of an L-shaped region with a specified spacing.
    The boundary consists of 6 segments: 
    bottom, lower-right corner, right-top edge, right, top, and left.
    """
    # Segment lengths
    length_1 = 1  # Bottom edge (0, 0) to (1, 0)
    length_2 = 0.5  # Lower-right edge (1, 0) to (1, 0.5)
    length_3 = 0.5  # Right-top edge (1, 0.5) to (0.5, 0.5)
    length_4 = 0.5  # Right edge (0.5, 0.5) to (0.5, 1)
    length_5 = 0.5  # Top edge (0.5, 1) to (0, 1)
    length_6 = 1  # Left edge (0, 1) to (0, 0)

    # Calculate number of points for each segment
    num_points_1 = int(length_1 / target_spacing) + 1
    num_points_2 = int(length_2 / target_spacing) + 1
    num_points_3 = int(length_3 / target_spacing) + 1
    num_points_4 = int(length_4 / target_spacing) + 1
    num_points_5 = int(length_5 / target_spacing) + 1
    num_points_6 = int(length_6 / target_spacing) + 1

    # Generate points for each segment
    x1, y1 = torch.linspace(0, 1, num_points_1), torch.zeros(num_points_1)
    x2, y2 = torch.full((num_points_2,), 1), torch.linspace(0, 0.5, num_points_2)
    x3, y3 = torch.linspace(1, 0.5, num_points_3), torch.full((num_points_3,), 0.5)
    x4, y4 = torch.full((num_points_4,), 0.5), torch.linspace(0.5, 1, num_points_4)
    x5, y5 = torch.linspace(0.5, 0, num_points_5), torch.ones(num_points_5)
    x6, y6 = torch.zeros(num_points_6), torch.linspace(1, 0, num_points_6)

    # Combine points from all segments and remove duplicates
    boundary_points = torch.cat([
        torch.stack((x1, y1), dim=1),
        torch.stack((x2, y2), dim=1),
        torch.stack((x3, y3), dim=1),
        torch.stack((x4, y4), dim=1),
        torch.stack((x5, y5), dim=1),
        torch.stack((x6, y6), dim=1)
    ], dim=0)
    
    return torch.unique(boundary_points, dim=0)

def generate_l_shape_inner_points(num_points):
    """
    Generate uniformly distributed points inside an L-shaped region.
    """
    x, y = np.linspace(0, 1, num_points), np.linspace(0, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    # Retain points inside the L-shaped region
    inside_l_shape = (grid_points[:, 0] <= 0.5) | (grid_points[:, 1] <= 0.5)
    return torch.tensor(grid_points[inside_l_shape], dtype=torch.float32)

boundary = generate_l_shape_boundary_points(1 / (num_points - 1))

class LaplaceActivation(nn.Module):
    """Custom activation function inspired by Laplace equation solutions."""
    def __init__(self, a, A, B, C, D):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))  # Trainable parameter
        self.A, self.B = nn.Parameter(torch.tensor(A)), nn.Parameter(torch.tensor(B))
        self.C, self.D = nn.Parameter(torch.tensor(C)), nn.Parameter(torch.tensor(D))

    def forward(self, x):
        xx, yy = x[:, 0], x[:, 1]
        return (self.A * torch.cos(self.a * xx) + self.B * torch.sin(self.a * xx)) * \
               (self.C * torch.cosh(self.a * yy) + self.D * torch.sinh(self.a * yy))

class LaplaceNN(nn.Module):
    """Neural network with custom Laplace-inspired activation functions."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([
            LaplaceActivation(torch.randn(1).item(), torch.randn(1).item(),
                              torch.randn(1).item(), torch.randn(1).item(),
                              torch.randn(1).item())
            for _ in range(10)
        ])
        self.output = nn.Linear(10, 1)  # Fully connected output layer
        nn.init.normal_(self.output.weight, mean=0.0, std=1)
        nn.init.normal_(self.output.bias, mean=0.0, std=1)

    def forward(self, x):
        outputs = torch.cat([layer(x).unsqueeze(1) for layer in self.hidden], dim=1)
        return self.output(outputs)

def generate_training_data(num_networks, num_points):
    """
    Generate training data using Laplace-based neural networks.
    """
    all_data = []
    points = generate_l_shape_inner_points(num_points)
    
    for _ in range(num_networks):
        net = LaplaceNN()
        net.eval()
        x, y = points[:, 0:1].requires_grad_(True), points[:, 1:2].requires_grad_(True)
        u = net(torch.cat((x, y), dim=1))
        u_val = u.view(1, -1)
        u_b = net(boundary).view(1, -1)
        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())
    return np.vstack(all_data)

def save_training_data(data, file_path):
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/MADlaplace2D1L_{num_functions, num_points}.txt"
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
print(f"Training data saved to {file_path}.")
print(f"Time taken: {end_time - start_time:.2f} seconds.")
