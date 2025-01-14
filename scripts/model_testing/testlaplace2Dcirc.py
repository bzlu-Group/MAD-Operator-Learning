import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Constants for number of functions, points, and model/test items
num_functions = 2000
num_points = 51
model_item = 1  # Model option (for MAD model)
test_item = 1  # Test data set option

# Paths to test data and pre-trained model
test_file_path = f"data/MADlaplace2D{test_item}circ_(200, 51).txt"  # Path to the test dataset
#model_path = f"models/MADlaplace2D{model_item}circ_{num_functions,num_points}.pth"  # Path to the pre-trained MAD model
model_path = f"models/PINNlaplace2Dcirc_{num_functions,num_points}.pth"  # Alternative: PINN-based model

# Create Dataset class similar to training dataset
class TestDataset(Dataset):
    def __init__(self, file_path):
        # Read data from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line into a list of floats and store
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the number of data samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return the data sample at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test dataset
test_dataset = TestDataset(test_file_path)
batch_size = 200  # Set the batch size based on the size of the test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Define the trunk network
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
        
        # This tensor holds the weights for the last layer to compute the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the combined output using a dot product of the branch and trunk outputs
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u = linear_branch_DeepONet(100, 2, 110).to(device)  
u.load_state_dict(torch.load(model_path, weights_only=True))
u.eval()  # Set the model to evaluation mode

# Define relative L2 error calculation
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

# Define relative L_infinity error calculation
def relative_l_inf_error(u_pred, u_true):
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

# Initialize variables to accumulate errors and batch count
total_relative_l2_error = 0.0
total_relative_l_inf_error = 0.0
num_batches = 0

# Function to generate points uniformly distributed inside the unit circle
def generate_circle_inner_points(num_points):
    # Calculate grid step size
    h = 2 / (num_points - 1)
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter points inside the unit circle, keeping only those where x^2 + y^2 <= 1
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_circle = np.sum(grid_points**2, axis=1) <= 1
    num_inside_circle = np.sum(inside_circle)
    print(f"Number of points inside the unit circle: {num_inside_circle}")
    grid_points = grid_points[np.sum(grid_points**2, axis=1) <= 1]  # Keep only points inside the circle
    
    # Convert numpy array to PyTorch tensor
    circle_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return circle_inner_points

# Generate trunk data points inside the unit circle
trunk = generate_circle_inner_points(num_points).to(device)

# No gradient calculation needed during evaluation
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        inputs = batch.to(device)
        
        # Extract branch and trunk inputs from the batch
        num_rows = inputs.size(0)
        trunk_batch = trunk.repeat(num_rows, 1)
        u_true_batch = inputs[:, 100:].reshape(-1, 1).to(device)  # Reshape instead of view
        branch_batch = inputs[:, 0:100].unsqueeze(1).repeat(1, 1957, 1).reshape(-1, 100).to(device)
          
        # Make predictions using the model
        u_pred_batch = u(branch_batch, trunk_batch)

        # Calculate the relative L2 and L_inf errors for the batch
        batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
        batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
        
        # Accumulate errors
        total_relative_l2_error += batch_relative_l2_error.item()
        total_relative_l_inf_error += batch_relative_l_inf_error.item()
        num_batches += 1

# Compute the mean relative L2 and L_inf errors
mean_relative_l2_error = total_relative_l2_error / num_batches
mean_relative_l_inf_error = total_relative_l_inf_error / num_batches

# Print the results
print(f"Mean Relative L2 Error: {mean_relative_l2_error:.2e}")
print(f"Mean Relative L_inf Error: {mean_relative_l_inf_error:.2e}")
