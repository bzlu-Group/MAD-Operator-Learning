import torch
import torch.nn as nn
import numpy as np
import time

# Parameters
num_functions = 200  # Number of functions to generate
num_points = 51  

def generate_circle_points(num_points):
    """Generate evenly spaced points along the boundary of the unit circle."""
    theta = torch.linspace(0, 2 * np.pi, num_points)  # Angles from 0 to 2π
    x = torch.cos(theta)  # x-coordinates on the circle
    y = torch.sin(theta)  # y-coordinates on the circle
    circle_points = torch.stack((x, y), dim=1)
    return circle_points

def generate_circle_inner_points(num_points):
    """Generate grid points inside the unit circle."""
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter points outside the unit circle
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_circle = np.sum(grid_points ** 2, axis=1) <= 1  # Condition: x^2 + y^2 ≤ 1
    num_inside_circle = np.sum(inside_circle)
    print(f"Number of points inside the unit circle: {num_inside_circle}")
    grid_points = grid_points[inside_circle]  # Keep points inside the circle

    # Convert numpy array to PyTorch tensor
    circle_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return circle_inner_points

boundary = generate_circle_points(100)  # Generate 100 points along the unit circle boundary

class LaplaceActivation(nn.Module):
    """Custom activation function inspired by Laplace equation solutions."""
    def __init__(self, a, A, B, C, D):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))  # Trainable parameter
        self.A = nn.Parameter(torch.tensor(A))
        self.B = nn.Parameter(torch.tensor(B))
        self.C = nn.Parameter(torch.tensor(C))
        self.D = nn.Parameter(torch.tensor(D))

    def forward(self, x):
        """Apply the activation function based on the input x."""
        xx = x[:, 0]
        yy = x[:, 1]
        return (self.A * torch.cos(self.a * xx) + self.B * torch.sin(self.a * xx)) * \
               (self.C * torch.cosh(self.a * yy) + self.D * torch.sinh(self.a * yy))

class LaplaceNN(nn.Module):
    """Neural network with custom Laplace-inspired activation functions."""
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
        """Forward pass through the network."""
        outputs = torch.cat([h(x).unsqueeze(1) for h in self.hidden], dim=1)
        return self.output(outputs)
    
def generate_training_data(num_networks, num_points):
    """Generate training data by computing solutions for the Laplace equation."""
    all_data = []
    points = generate_circle_inner_points(num_points)  # Points inside the unit circle

    for _ in range(num_networks):
        net = LaplaceNN()
        net.eval()  # Set the network to evaluation mode
        x = points[:, 0:1].requires_grad_(True)  # Enable gradient computation for x
        y = points[:, 1:2].requires_grad_(True)  # Enable gradient computation for y
        u = net(torch.cat((x, y), dim=1))
        u_val = u.view(1, -1)
        u_b = net(boundary).view(1, -1)
        data_row = torch.cat((u_b, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value  # Normalize the data row
        all_data.append(normalized_vector.detach().numpy())
    return np.vstack(all_data)

def save_training_data(data, file_path):
    """Save the training data to a file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Main execution
start_time = time.time()
file_path = f"data/MADlaplace2D1circ_{num_functions, num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
