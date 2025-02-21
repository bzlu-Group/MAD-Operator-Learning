import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

num_functions = 2000
num_points = 51
k = 100  # △u + ku = 0, Helmholtz equation with k as the coefficient
test_item = 2  # Select test dataset (1 for dataset 1, 2 for dataset 2)
model_item = 2  # 1 for PINN-based model, 2 for MAD model
# Path for test dataset
if test_item == 1:
    test_file_path = f"data/MAD{k}helmholtz_withoutsource_2D_(200, 51).txt"
else:
    test_file_path = f"data/TNM{k}helmholtz_withoutsource_2D_(200, 51).txt"  

if model_item == 1:
    model_path = f"models/PINN{k}helmholtzwithoutsource2D_{num_functions,num_points}_1_1.pth" 
else:
    model_path = f"models/MAD{k}helmholtzwithoutsource2D_{num_functions,num_points}_1_1.pth" 

# Create Dataset class, similar to the training dataset
class TestDataset(Dataset):
    def __init__(self, file_path):
        # Read the file data
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line into a float and store it
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return a specific data point
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test dataset
test_dataset = TestDataset(test_file_path)
batch_size = 50  # Set an appropriate batch size based on the dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the linear branch network for the DeepONet model
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
        
        # A tensor to store the last layer weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Combine the outputs of the branch and trunk networks
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)  
u.load_state_dict(torch.load(model_path, weights_only=True))
u.eval()  # Set the model to evaluation mode

# Define the relative L2 error function
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

# Define the relative L_inf error function
def relative_l_inf_error(u_pred, u_true):
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

# Initialize variables to accumulate errors
total_relative_l2_error = 0.0
total_relative_l_inf_error = 0.0
num_batches = 0

# Define the grid resolution (h)
h = 1 / (num_points - 1)
# Generate the grid points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert the grid points to a PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Disable gradient computation for evaluation
with torch.no_grad():  
    for i, batch in enumerate(test_loader):
        inputs = batch.to(device)
        
        # Extract the branch and trunk network inputs
        num_rows = inputs.size(0)
        trunk_batch = trunk.repeat(num_rows, 1)
        u_true_batch = inputs[:, 4 * (num_points - 1):].reshape(-1, 1).to(device)  # Use reshape instead of view
        branch_batch = inputs[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).reshape(-1, 4 * (num_points - 1)).to(device)
          
        # Predict the output using the model
        u_pred_batch = u(branch_batch, trunk_batch)

        # Calculate the relative L2 error for the current batch
        batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
        batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
        total_relative_l2_error += batch_relative_l2_error.item()
        total_relative_l_inf_error += batch_relative_l_inf_error.item()
        num_batches += 1

# Compute the average relative L2 and L_inf errors
mean_relative_l2_error = total_relative_l2_error / num_batches
mean_relative_l_inf_error = total_relative_l_inf_error / num_batches
print(f"Mean Relative L2 Error: {mean_relative_l2_error:.2e}")
print(f"Mean Relative L_inf Error: {mean_relative_l_inf_error:.2e}")
