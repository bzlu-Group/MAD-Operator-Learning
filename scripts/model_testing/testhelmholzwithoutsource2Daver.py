import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Parameter settings
num_networks = 2000
num_points = 51
k = 100
model_item = 1
test_item = 1

# Test dataset path
if test_item == 1:
    test_file_path = f"data/MAD{k}helmholtz_withoutsource_2D_(200, 51).txt" 
else:
    test_file_path = f"data/TNM{k}helmholtz_withoutsource_2D_(200, 51).txt" 

# Define model paths based on the value of 'model_item'
if model_item == 2:
    model_paths = [
        f"models/MAD{k}helmholtzwithoutsource2D_{num_networks,num_points}_1_1.pth",
        f"models/MAD{k}helmholtzwithoutsource2D_{num_networks,num_points}_1_2.pth",
        f"models/MAD{k}helmholtzwithoutsource2D_{num_networks,num_points}_2_1.pth",
        f"models/MAD{k}helmholtzwithoutsource2D_{num_networks,num_points}_2_2.pth"
    ]
else:
    model_paths = [
        f"models/PINN{k}helmholtzwithoutsource2D_{num_networks,num_points}_1_1.pth",
        f"models/PINN{k}helmholtzwithoutsource2D_{num_networks,num_points}_1_2.pth",
        f"models/PINN{k}helmholtzwithoutsource2D_{num_networks,num_points}_2_1.pth",
        f"models/PINN{k}helmholtzwithoutsource2D_{num_networks,num_points}_2_2.pth"
    ]

# Define the Dataset class for loading test data
class TestDataset(Dataset):
    def __init__(self, file_path):
        # Read data from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line to a list of floats and store
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Get the data point at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test dataset
test_dataset = TestDataset(test_file_path)
batch_size = 50  # Set an appropriate batch size based on the dataset size
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model class
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
        
        # This tensor will hold the last layer's weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass inputs through the branch and trunk networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Combine outputs using dot product
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Define the relative L2 error calculation
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

# Define the relative L∞ error calculation
def relative_l_inf_error(u_pred, u_true):
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

# Initialize lists to store errors
relative_l2_errors = []
relative_l_inf_errors = []

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate grid points
h = 1 / (num_points - 1)
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Iterate over each model
for model_path in model_paths:
    # Load the model
    u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)
    u.load_state_dict(torch.load(model_path, weights_only=True))
    u.eval()  # Set the model to evaluation mode

    total_relative_l2_error = 0.0
    total_relative_l_inf_error = 0.0
    num_batches = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for i, batch in enumerate(test_loader):
            inputs = batch.to(device)
            
            # Extract inputs for the branch and trunk networks
            num_rows = inputs.size(0)
            trunk_batch = trunk.repeat(num_rows, 1)
            u_true_batch = inputs[:, 4 * (num_points - 1):].reshape(-1, 1).to(device)  # Use reshape instead of view
            branch_batch = inputs[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).reshape(-1, 4 * (num_points - 1)).to(device)
            
            # Predict values
            u_pred_batch = u(branch_batch, trunk_batch)

            # Calculate relative L2 and L∞ errors for the current batch
            batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
            batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
            total_relative_l2_error += batch_relative_l2_error.item()
            total_relative_l_inf_error += batch_relative_l_inf_error.item()
            num_batches += 1

    # Calculate the average errors for the current model
    mean_relative_l2_error = total_relative_l2_error / num_batches
    mean_relative_l_inf_error = total_relative_l_inf_error / num_batches

    # Append the current model's errors to the lists
    relative_l2_errors.append(mean_relative_l2_error)
    relative_l_inf_errors.append(mean_relative_l_inf_error)

# Calculate the mean and standard deviation of the errors
avg_relative_l2_error = np.mean(relative_l2_errors)
std_relative_l2_error = np.std(relative_l2_errors)
avg_relative_l_inf_error = np.mean(relative_l_inf_errors)
std_relative_l_inf_error = np.std(relative_l_inf_errors)

# Output the results in scientific notation with 2 decimal places
print(f"Average Relative L2 Error: {avg_relative_l2_error:.2e}(±{std_relative_l2_error:.2e})")
print(f"Average Relative L∞ Error: {avg_relative_l_inf_error:.2e}(±{std_relative_l_inf_error:.2e})")