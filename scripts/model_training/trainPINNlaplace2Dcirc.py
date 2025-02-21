import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

# Number of functions and points in the grid
num_functions = 2000
num_points = 51
checkpoint_path = f"models/PINNlaplace2Dcirc_checkpoint_1_1.pth"  # Path to save/load the model checkpoint

# Dataset class to load data
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Process data into a list of floats
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return a tensor of the data at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()  # Clear GPU memory cache

# Determine whether to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_circle_inner_points(num_points):
    """
    Generates points uniformly distributed inside the unit circle.

    Args:
        num_points (int): Number of grid points in the domain.

    Returns:
        torch.Tensor: Points inside the unit circle as a tensor.
    """
    h = 2 / (num_points - 1)  # Grid step size
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter out points outside the unit circle (x^2 + y^2 <= 1)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_circle = np.sum(grid_points**2, axis=1) <= 1
    num_inside_circle = np.sum(inside_circle)
    print(f"Number of points inside the unit circle: {num_inside_circle}")
    grid_points = grid_points[inside_circle]  # Keep only points inside the circle
    
    # Convert to PyTorch tensor
    circle_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return circle_inner_points

# Generate points inside the unit circle and move them to the GPU
trunk = generate_circle_inner_points(num_points).to(device)

def generate_circle_points(num_points):
    """
    Generates points on the unit circle boundary.

    Args:
        num_points (int): Number of points on the boundary.

    Returns:
        torch.Tensor: Points on the boundary of the unit circle as a tensor.
    """
    theta = torch.linspace(0, 2 * np.pi, num_points)
    x = torch.cos(theta)  # x-coordinates on the unit circle
    y = torch.sin(theta)  # y-coordinates on the unit circle
    circle_points = torch.stack((x, y), dim=1)  # Stack x and y coordinates
    return circle_points

# Generate boundary points on the unit circle and move them to the GPU
boundpoints = generate_circle_points(2 * num_points - 2).to(device)

class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        """
        Initializes the neural network model for DeepONet with separate branch and trunk networks.

        Args:
            branch_features (int): Number of features in the branch input.
            trunk_features (int): Number of features in the trunk input.
            common_features (int): Number of common features in the hidden layers.
        """
        super(linear_branch_DeepONet, self).__init__()
        
        # Branch network: Takes input features and transforms them
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Trunk network: Takes input points and transforms them
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
        
        # Learnable weights for the final dot product computation
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        """
        Forward pass through the branch and trunk networks.

        Args:
            branch_input (torch.Tensor): Input tensor for the branch network.
            trunk_input (torch.Tensor): Input tensor for the trunk network.

        Returns:
            torch.Tensor: The output of the network after combining branch and trunk outputs.
        """
        # Pass inputs through branch and trunk networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the dot product between branch and trunk outputs with learnable weights
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output
    
# Initialize the model and wrap it in DataParallel for multi-GPU usage
u = linear_branch_DeepONet(100, 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Using DataParallel to utilize multiple GPUs if available

# Loss function (Mean Squared Error)
loss = torch.nn.MSELoss().to(device)

def grad(u, x):
    """
    Computes the gradient of the output u with respect to the input x.

    Args:
        u (torch.Tensor): The model output.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The gradient of u with respect to x.
    """
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

def l_i(u, inputs1):
    """
    Computes the interior loss for the PDE residual.

    Args:
        u (nn.Module): The neural network model.
        inputs1 (torch.Tensor): Input data for the interior loss.

    Returns:
        torch.Tensor: The computed loss for interior points.
    """
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)  # Repeat trunk points for each input batch
    x = trunk_batch[:, 0:1].requires_grad_(True)
    y = trunk_batch[:, 1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    branch_batch = inputs1[:, :100].unsqueeze(1).repeat(1, 1957, 1).view(-1, 100).to(device)
    
    # Predict the solution using the model
    upred_batch = u(branch_batch, trunk_batch)
    
    # Compute the second derivatives (Laplace operator)
    upp = grad(grad(upred_batch, x), x) + grad(grad(upred_batch, y), y)

    # Compute the MSE loss between predicted and true values (zero in this case)
    l = loss(upp, torch.zeros_like(upp))
    return l

def l_b(u, inputs1):
    """
    Computes the boundary loss for the model.

    Args:
        u (nn.Module): The neural network model.
        inputs1 (torch.Tensor): Input data for the boundary loss.

    Returns:
        torch.Tensor: The computed loss for boundary points.
    """
    num_rows = inputs1.size(0)
    trunk_batch = boundpoints.repeat(num_rows, 1)  # Repeat boundary points for each input batch
    
    boundvalue = inputs1[:, :100].reshape(-1, 1).to(device)
    branch_batch = inputs1[:, :100].unsqueeze(1).repeat(1, 100, 1).view(-1, 100).to(device)
    
    # Predict the solution using the model
    upred = u(branch_batch, trunk_batch)

    # Compute the MSE loss between predicted and boundary values
    l = loss(upred, boundvalue)
    return l

# Path to the dataset file
file_path = f"data/PINNlaplace2Dcirc_{num_functions, num_points}_1.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching the dataset
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer (Adam) and learning rate scheduler (ReduceLROnPlateau)
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load model checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Use .module for DataParallel
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    minloss = checkpoint['min_loss']
    loss0 = checkpoint['loss_records']
    elapsed_time = checkpoint['elapsed_time']
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with minloss {minloss}.")
else:
    start_epoch = 0
    minloss = 1
    loss0 = []
    elapsed_time = 0
    print("No checkpoint found, starting from scratch.")

# Training loop parameters
epoches = 200000
model_path = f"models/PINNlaplace2Dcirc_{num_functions, num_points}_1_1.pth"

start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        # Compute the combined loss (interior loss + boundary loss)
        loss_function = 0.9 * l_i(u, inputs) + 0.1 * l_b(u, inputs)
        l = loss_function.item()
        
        # Save the model if the current loss is the minimum
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save the model using .module
            minloss = l
        
        # Backpropagate and optimize
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    # Print training progress
    print(f"Epoch [{epoch+1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        end_time = time.time()
        elapsed_time += end_time - start_time
        start_time = end_time
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': u.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': minloss,
            'loss_records': loss0,
            'elapsed_time': elapsed_time
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1} with elapsed time {elapsed_time:.2f} seconds.")

# Print total training time
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")
