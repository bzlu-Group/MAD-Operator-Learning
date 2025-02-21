import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Constants for number of functions, points, and model/test items
num_functions = 2000
num_points = 51
model_item = 2
test_item = 2

# Paths to test data and pre-trained model
test_file_path = f"data/MADlaplace2D{test_item}L_(200, 51).txt"  # Path to the test dataset
#model_path = f"models/MADlaplace2D{model_item}L_{num_functions,num_points}.pth"  # Path to the pre-trained model
model_path = f"models/PINNlaplace2DL_{num_functions,num_points}_1_1.pth"  # Another model option

# Create Dataset class similar to training dataset
class TestDataset(Dataset):
    def __init__(self, file_path):
        # Read the file data
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line into a float and store
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the number of data entries in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return the data entry at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test data
test_dataset = TestDataset(test_file_path)
batch_size = 200  # Set an appropriate batch size based on the size of the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch net
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Define the trunk net
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
        
        # This tensor will hold the last layer's weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u = linear_branch_DeepONet(200, 2, 110).to(device)  
u.load_state_dict(torch.load(model_path, weights_only=True))
u.eval()  # Set to evaluation mode

# Define relative L2 error computation
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

def relative_l_inf_error(u_pred, u_true):
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

total_relative_l2_error = 0.0
total_relative_l_inf_error = 0.0
num_batches = 0

def generate_l_shape_inner_points(num_points):
    """
    Generate points uniformly distributed within an L-shaped region
    """
    h = 2 / (num_points - 1)  # Grid step size
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter out points from the top-right quarter of the square
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    # Retain points inside the L-shaped region
    inside_l_shape = (grid_points[:,0] <= 0.5) | (grid_points[:,1] <= 0.5)
    grid_points = grid_points[inside_l_shape]
    
    # Convert numpy array to PyTorch tensor
    l_shape_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return l_shape_inner_points

# Convert numpy array to PyTorch tensor
trunk = generate_l_shape_inner_points(num_points).to(device)

with torch.no_grad():  # No need to compute gradients during evaluation
    for i, batch in enumerate(test_loader):
        inputs = batch.to(device)
        
        # Extract branch and trunk network inputs
        num_rows = inputs.size(0)
        trunk_batch = trunk.repeat(num_rows, 1)
        u_true_batch = inputs[:, 200:].reshape(-1, 1).to(device)  # Use reshape instead of view
        branch_batch = inputs[:, 0:200].unsqueeze(1).repeat(1, 1976, 1).reshape(-1, 200).to(device)
          
        # Make predictions
        u_pred_batch = u(branch_batch, trunk_batch)

        # Compute relative L2 error for each batch
        batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
        batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
        total_relative_l2_error += batch_relative_l2_error.item()
        total_relative_l_inf_error += batch_relative_l_inf_error.item()
        num_batches += 1

# Compute average relative L2 error
mean_relative_l2_error = total_relative_l2_error / num_batches
mean_relative_l_inf_error = total_relative_l_inf_error / num_batches
print(f"Mean Relative L2 Error: {mean_relative_l2_error:.2e}")
print(f"Mean Relative L_inf Error: {mean_relative_l_inf_error:.2e}")
