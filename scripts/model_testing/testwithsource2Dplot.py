import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# Configuration parameters
num_functions = 2000
num_points = 51
k = 1  # △u + k*u = f; k = 0, 1 ,10, 100
n = 200  # Grid resolution for visualization

# Define the test function
def testfun(point):
    x = point[:, 0:1]
    y = point[:, 1:2]
    # f = x + y
    # f = x**2 + y**2 - x*y
    f = torch.exp(x**2 + y**2) 
    return f

# Laplace operator
def laplace(fun, point):
    point.requires_grad_(True)
    x = point[:, 0:1]
    y = point[:, 1:2]
    f = fun(torch.cat((x, y), dim=1))
    # Compute gradients
    grad_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    grad_y = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    # Compute second-order gradients
    grad_x2 = torch.autograd.grad(grad_x + x - x, x, grad_outputs=torch.ones_like(grad_x), create_graph=True)[0]
    grad_y2 = torch.autograd.grad(grad_y + y - y, y, grad_outputs=torch.ones_like(grad_y), create_graph=True)[0]
    lap = grad_x2 + grad_y2 + k * f  # Laplace operator with k*u term
    return lap.detach()  # Detach the result from computation graph

# Generate points on a square boundary
def generate_square_points(num_points):
    # Determine the number of points per edge (excluding the last point on each edge)
    points_per_edge = num_points
    # Generate points for four edges
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge - 1)), dim=1)
    edge2 = torch.stack((torch.ones(points_per_edge - 1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge - 1)), dim=1)
    edge4 = torch.stack((torch.zeros(points_per_edge - 1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

# Generate grid points within a square
def generate_grid_points(num_points):
    x = torch.linspace(0, 1, num_points)
    y = torch.linspace(0, 1, num_points)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return grid_points, grid_x, grid_y

# Define the DeepONet model
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        self.branch_net = nn.Sequential(nn.Linear(branch_features, 1000, bias=False))
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, 1000),
        )
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        # Combine outputs from branch and trunk
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Define the PoissonONet model
class possionONet(nn.Module):
    def __init__(self, branch1_features, branch2_features, trunk_features, common_features1, common_features2):
        super(possionONet, self).__init__()
        self.net1 = linear_branch_DeepONet(branch1_features, trunk_features, common_features1)
        self.net2 = linear_branch_DeepONet(branch2_features, trunk_features, common_features2)

    def forward(self, branch1_input, branch2_input, trunk_input):
        output1 = self.net1(branch1_input, trunk_input)
        output2 = self.net2(branch2_input, trunk_input)
        # Combine outputs from both nets
        combined_output = output1 + output2
        return combined_output

# Load the trained model
def load_model(model_path):
    model = possionONet(4 * (num_points - 1), num_points * num_points, 2, 90, 110)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Test the model
def test_model(model, branch1, branch2, points):
    branch1 = branch1.repeat((n + 1) * (n + 1), 1)  # Repeat branch input for all points
    branch2 = branch2.repeat((n + 1) * (n + 1), 1)  # Repeat branch input for all points
    u_pred = model(branch1, branch2, points)
    return u_pred

# Main testing and plotting process
def run_test_and_plot():
    # Prepare data
    points, grid_x, grid_y = generate_grid_points(n + 1)
    interior_points, _, _ = generate_grid_points(num_points)
    boundary_points = generate_square_points(num_points)
    
    # Compute branch inputs
    branch1 = testfun(boundary_points).view(1, -1)
    branch2 = laplace(testfun, interior_points).view(1, -1)
    u_true = testfun(points)  # True solution
    
    # Load MAD and PINN models
    if k == 0:
        mad_model_path = f"models/MADpoisson2D_{num_functions,num_points}.pth"
        pinn_model_path = f"models/PINNpoisson2D_{num_functions,num_points}.pth"
    else:
        mad_model_path = f"models/MAD{k}helmholtz2D_{num_functions,num_points}.pth"
        pinn_model_path = f"models/PINN{k}helmholtz2D_{num_functions,num_points}.pth"
    mad_model = load_model(mad_model_path)
    pinn_model = load_model(pinn_model_path)
    
    # Get predictions
    u_pred_mad = test_model(mad_model, branch1, branch2, points)
    u_pred_pinn = test_model(pinn_model, branch1, branch2, points)
    
    # Convert predictions to numpy arrays
    u_pred_mad_np = u_pred_mad.view(n + 1, n + 1).detach().numpy()
    u_pred_pinn_np = u_pred_pinn.view(n + 1, n + 1).detach().numpy()
    u_true_np = u_true.view(n + 1, n + 1).detach().numpy()
    
    # Compute errors
    error_mad = np.abs(u_pred_mad_np - u_true_np)
    error_pinn = np.abs(u_pred_pinn_np - u_true_np)
    
    # Compute relative L2 errors
    rel_error_mad = np.linalg.norm(error_mad) / np.linalg.norm(u_true_np)
    rel_error_pinn = np.linalg.norm(error_pinn) / np.linalg.norm(u_true_np)
    
    # Plot results
    plt.figure(figsize=(18, 12))
    
    # First row: True solution, MAD prediction, MAD error
    plt.subplot(2, 3, 1)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), u_true_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('True Solution', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    plt.subplot(2, 3, 2)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), u_pred_mad_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'(Relative L2 Error: {rel_error_mad:.2e})\nMAD Prediction', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    plt.subplot(2, 3, 3)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), error_mad, levels=50, cmap='inferno')
    plt.colorbar()
    plt.title('MAD Error', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Second row: True solution, PINN prediction, PINN error
    plt.subplot(2, 3, 4)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), u_true_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('True Solution', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)

    plt.subplot(2, 3, 5)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), u_pred_pinn_np, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'(Relative L2 Error: {rel_error_pinn:.2e})\nPINN Prediction', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    plt.subplot(2, 3, 6)
    plt.contourf(grid_x.numpy(), grid_y.numpy(), error_pinn, levels=50, cmap='inferno')
    plt.colorbar()
    plt.title('PINN Error', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.show()

# Run the test
run_test_and_plot()

