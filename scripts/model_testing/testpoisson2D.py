import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Constants for the number of functions, points, and test/model selection
num_functions = 2000
num_points = 51
test_item = 1  # Select test dataset 1 or 2
model_item = 1  # 1 for PINN-based model, 2 for MAD model

# Set the test data file path based on the test item choice
if test_item == 1:
    test_file_path = f"data/MADpoisson2D_(200, 51).txt"  # Path for test data set 1
else:
    test_file_path = f"data/TNMpoisson2D_(200, 51).txt"  # Path for test data set 2

# Set the model path based on the model item choice
if model_item == 1:
    model_path = f"models/PINNpoisson2D_{num_functions,num_points}.pth"  # Path for PINN-based model
else:
    model_path = f"models/MADpoisson2D_{num_functions,num_points}.pth"  # Path for MAD model

# Dataset class for loading test data
class TestDataset(Dataset):
    def __init__(self, file_path):
        """
        Reads test data from a file where each line contains a set of values.
        Each line is split into individual floating-point numbers and stored.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Convert each line into a list of floating-point values
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a sample at the given index and returns it as a tensor.
        """
        return torch.tensor(self.data[index], dtype=torch.float32)

# Load the test dataset and create a DataLoader for batching
test_dataset = TestDataset(test_file_path)
batch_size = 200  # Adjust batch size according to the test dataset size
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the linear branch DeepONet model
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network (without bias)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Define the trunk network (with several layers)
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
        
        # This parameter holds the weights for the last layer's dot product operation
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        """
        Pass the inputs through their respective networks and compute the output.
        """
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Combine the outputs of the branch and trunk networks using a weighted sum
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Define the overall PoissonONet model, consisting of two branch networks
class possionONet(nn.Module):
    def __init__(self, branch1_features, branch2_features, trunk_features, common_features1, common_features2):
        super(possionONet, self).__init__()
        # Define the two branch networks
        self.net1 = linear_branch_DeepONet(branch1_features, trunk_features, common_features1)
        self.net2 = linear_branch_DeepONet(branch2_features, trunk_features, common_features2)

    def forward(self, branch1_input, branch2_input, trunk_input):
        """
        Pass inputs through their respective networks and combine their outputs.
        """
        output1 = self.net1(branch1_input, trunk_input)
        output2 = self.net2(branch2_input, trunk_input)
        
        # Combine the outputs from both branch networks
        combined_output = output1 + output2
        return combined_output

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u = possionONet(4 * (num_points - 1), num_points * num_points, 2, 90, 110).to(device)  # Create model and move it to the device (GPU/CPU)
u.load_state_dict(torch.load(model_path, weights_only=True))  # Load model weights
u.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)

# Function to calculate relative L2 error
def relative_l2_error(pred, true):
    """
    Compute the relative L2 error between predicted and true values.
    """
    return torch.norm(pred - true) / torch.norm(true)

# Function to calculate relative L∞ error
def relative_l_inf_error(u_pred, u_true):
    """
    Compute the relative L∞ error (maximum absolute error) between predicted and true values.
    """
    abs_error = torch.abs(u_pred - u_true)
    l_infinity_error = torch.max(abs_error)
    relative_error = l_infinity_error / torch.max(torch.abs(u_true))
    return relative_error

# Variables to accumulate total errors across all batches
total_relative_l2_error = 0.0
total_relative_l_inf_error = 0.0
num_batches = 0

h = 1 / (num_points - 1)  # Grid spacing
# Generate grid points for the Poisson problem
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert the numpy array to a PyTorch tensor and move it to the device
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Disable gradient computation during evaluation
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        inputs = batch.to(device)
        
        # Prepare the branch and trunk inputs for the model
        num_rows = inputs.size(0)
        trunk_batch = trunk.repeat(num_rows, 1)  # Repeat trunk input for each data sample
        u_true_batch = inputs[:, 4 * (num_points - 1) + num_points * num_points:].reshape(-1, 1).to(device)  # True solution values
        branch1_batch = inputs[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).reshape(-1, 4 * (num_points - 1)).to(device)
        branch2_batch = inputs[:, 4 * (num_points - 1):4 * (num_points - 1) + num_points * num_points].unsqueeze(1).repeat(1, num_points * num_points, 1).reshape(-1, num_points * num_points).to(device)
          
        # Make predictions using the model
        u_pred_batch = u(branch1_batch, branch2_batch, trunk_batch)

        # Compute the relative L2 and L∞ errors for the current batch
        batch_relative_l2_error = relative_l2_error(u_pred_batch, u_true_batch)
        batch_relative_l_inf_error = relative_l_inf_error(u_pred_batch, u_true_batch)
        total_relative_l2_error += batch_relative_l2_error.item()  # Accumulate L2 errors
        total_relative_l_inf_error += batch_relative_l_inf_error.item()  # Accumulate L∞ errors
        num_batches += 1  # Increment the batch counter

# Calculate the average relative L2 and L∞ errors across all batches
mean_relative_l2_error = total_relative_l2_error / num_batches
mean_relative_l_inf_error = total_relative_l_inf_error / num_batches
# Print the results
print(f"Mean Relative L2 Error: {mean_relative_l2_error:.2e}")
print(f"Mean Relative L_inf Error: {mean_relative_l_inf_error:.2e}")
