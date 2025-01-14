import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

num_functions = 2000
num_points = 51
k = 100  # Constant k for the Helmholtz equation
test_item = 1  # Choose test dataset 1 or 2
model_item = 1  # 1 for PINN-based model, 2 for MAD model

# Set the path for the test dataset based on the selected test item
if test_item == 1:
    test_file_path = f"data/MAD{k}helmholtz2D_(200, 51).txt"
else:
    test_file_path = f"data/TNM{k}helmholtz2D_(200, 51).txt"

# Set the path for the model based on the selected model item
if model_item == 1:
    model_path = f"models/PINN{k}helmholtz2D_{num_functions,num_points}.pth"
else:
    model_path = f"models/MAD{k}helmholtz2D_{num_functions,num_points}.pth"  

# Create a Dataset class similar to the training dataset
class TestDataset(Dataset):
    def __init__(self, file_path):
        # Read the file data
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line to a list of floats and store
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the number of data samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return the data sample at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test dataset
test_dataset = TestDataset(test_file_path)
batch_size = 200  # Set an appropriate batch size based on the test dataset size
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
        
        # This tensor holds the last layer's weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the dot product of the branch and trunk outputs
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output
    
class possionONet(nn.Module):
    def __init__(self, branch1_features, branch2_features, trunk_features, common_features1, common_features2):
        super(possionONet, self).__init__()
        # Define the two branch networks
        self.net1 = linear_branch_DeepONet(branch1_features, trunk_features, common_features1)
        self.net2 = linear_branch_DeepONet(branch2_features, trunk_features, common_features2)

    def forward(self, branch1_input, branch2_input, trunk_input):
        # Pass the inputs through their respective networks
        output1 = self.net1(branch1_input, trunk_input)
        output2 = self.net2(branch2_input, trunk_input)
        
        # Combine the outputs from the two networks
        combined_output = output1 + output2
        return combined_output

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u = possionONet(4*(num_points-1), num_points*num_points, 2, 90, 110).to(device)  
u.load_state_dict(torch.load(model_path, weights_only=True))
u.eval()  # Set the model to evaluation mode

# Define the relative L2 error calculation
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

# Define the relative L_inf error calculation
def relative_l_inf_error(u_pred, u_true):
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

# Initialize variables for accumulating errors and batch count
total_relative_l2_error = 0.0
total_relative_l_inf_error = 0.0
num_batches = 0

h = 1 / (num_points - 1)  # Calculate the step size for the grid
# Generate the grid points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Convert the grid points to a PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

with torch.no_grad():  # No need to compute gradients during evaluation
    for i, batch in enumerate(test_loader):
        inputs = batch.to(device)
        
        # Extract the branch and trunk inputs from the batch
        num_rows = inputs.size(0)
        trunk_batch = trunk.repeat(num_rows, 1)
        u_true_batch = inputs[:, 4*(num_points-1)+num_points*num_points:].reshape(-1, 1).to(device)  # Use reshape instead of view
        branch1_batch = inputs[:, 0:4*(num_points-1)].unsqueeze(1).repeat(1, num_points*num_points, 1).reshape(-1, 4*(num_points-1)).to(device)
        branch2_batch = inputs[:, 4*(num_points-1):4*(num_points-1)+num_points*num_points].unsqueeze(1).repeat(1, num_points*num_points, 1).reshape(-1, num_points*num_points).to(device)
          
        # Make predictions
        u_pred_batch = u(branch1_batch, branch2_batch, trunk_batch)

        # Calculate the relative L2 error for the batch
        batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
        batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
        total_relative_l2_error += batch_relative_l2_error.item()
        total_relative_l_inf_error += batch_relative_l_inf_error.item()
        num_batches += 1

# Compute the average relative L2 error and L_inf error
mean_relative_l2_error = total_relative_l2_error / num_batches
mean_relative_l_inf_error = total_relative_l_inf_error / num_batches

# Print the results
print(f"Mean Relative L2 Error: {mean_relative_l2_error:.2e}")
print(f"Mean Relative L_inf Error: {mean_relative_l_inf_error:.2e}")
