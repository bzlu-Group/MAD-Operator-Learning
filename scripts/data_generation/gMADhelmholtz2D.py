import torch
import torch.nn as nn
import numpy as np
import time

# Parameters
num_functions = 2000
num_points = 51
k = 10  # Equation: △u+ku=f

def generate_square_points(num_points):
    """Generate points along the edges of a square boundary."""
    points_per_edge = num_points
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge-1)), dim=1)
    edge2 = torch.stack((torch.ones(points_per_edge-1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge-1)), dim=1)
    edge4 = torch.stack((torch.zeros(points_per_edge-1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

boundary = generate_square_points(num_points)

class CustomNetwork(nn.Module):
    """Custom neural network with sine activation functions."""
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights and biases with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.normal_(m.bias, mean=0, std=1)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_training_data(num_networks, num_points):
    """Generate training data by sampling from custom networks."""
    all_data = []
    h = 1 / (num_points - 1)
    x = np.arange(0, 1 + h, h)
    y = np.arange(0, 1 + h, h)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    points = torch.tensor(grid_points, dtype=torch.float32)

    for _ in range(num_networks):
        net = CustomNetwork()
        net.eval()
        x = points[:, 0:1].requires_grad_(True)
        y = points[:, 1:2].requires_grad_(True)
        u = net(torch.cat((x, y), dim=1))
        pux = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        puy = torch.autograd.grad(u, y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        ppuxx = torch.autograd.grad(pux, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        ppuyy = torch.autograd.grad(puy, y, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        laplace = ppuxx + ppuyy
        laplace = laplace.view(1, -1)
        u_val = u.view(1, -1)
        f = laplace + k * u_val
        u_b = net(boundary).view(1, -1)
        data_row = torch.cat((u_b, f, u_val), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())
    return np.vstack(all_data)

def save_training_data(data, file_path):
    """Save the generated training data to a file."""
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/MAD{k}helmholtz2D_{num_functions,num_points}.txt"
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
