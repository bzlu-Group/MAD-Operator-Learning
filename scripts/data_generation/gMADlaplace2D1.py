import torch
import torch.nn as nn
import numpy as np
import time

# Parameters
num_functions = 200  # Number of functions to generate
num_points = 51  # Number of points along one dimension

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

class LaplaceActivation(nn.Module):
    """Custom activation function based on Laplace equation solutions."""
    def __init__(self, a, A, B, C, D):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))  # Trainable parameter
        self.A = nn.Parameter(torch.tensor(A))
        self.B = nn.Parameter(torch.tensor(B))
        self.C = nn.Parameter(torch.tensor(C))
        self.D = nn.Parameter(torch.tensor(D))

    def forward(self, x):
        xx = x[:, 0]
        yy = x[:, 1]
        return (self.A * torch.cos(self.a * xx) + self.B * torch.sin(self.a * xx)) * \
               (self.C * torch.cosh(self.a * yy) + self.D * torch.sinh(self.a * yy))

class LaplaceNN(nn.Module):
    """Neural network with custom Laplace-based activations."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList()
        for _ in range(10):
            a = torch.randn(1).item()
            A = torch.randn(1).item()
            B = torch.randn(1).item()
            C = torch.randn(1).item()
            D = torch.randn(1).item()
            self.hidden.append(LaplaceActivation(a, A, B, C, D))
        
        self.output = nn.Linear(10, 1)  # Fully connected output layer
        nn.init.normal_(self.output.weight, mean=0.0, std=1)  # Initialize weights
        nn.init.normal_(self.output.bias, mean=0, std=1)  # Initialize bias

    def forward(self, x):
        outputs = torch.cat([h(x).unsqueeze(1) for h in self.hidden], dim=1)
        return self.output(outputs)

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
        net = LaplaceNN()
        net.eval()  # Set network to evaluation mode
        x = points[:, 0:1]
        y = points[:, 1:2]
        u = net(torch.cat((x, y), dim=1))
        u_val = u.view(1, -1)
        u_b = net(boundary).view(1, -1)
        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(u_b))
        normalized_vector = data_row / max_value  # Normalize to avoid large value ranges
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """Save training data to a file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Execution starts here
start_time = time.time()
file_path = f"data/MADlaplace2D1_{num_functions, num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
