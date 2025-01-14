import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

# Define constants for the number of functions and data points used during training
num_functions = 2000
num_points = 51
checkpoint_path = f"models/PINNlaplace2DL_checkpoint.pth"

# Dataset class to load training data
class Dataset(Dataset):
    def __init__(self, file_path):
        """
        Read data from the file and convert it to floating point numbers.
        Each line in the file represents one data sample.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        """
        Return the total number of data samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Return the data sample at the specified index as a PyTorch tensor.
        """
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()

# Check if GPU is available for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate boundary points for the L-shaped region
def generate_l_shape_boundary_points(target_spacing):
    """
    Generate boundary points for an L-shaped region, ensuring that the spacing between points 
    on each edge is `target_spacing`. The L-shaped region has 6 edges, and we calculate 
    how many points are needed for each edge.
    """
    length_1 = 1  # Length of the bottom edge (from (0, 0) to (1, 0))
    length_2 = 0.5  # Length of the right-bottom corner (from (1, 0) to (1, 0.5))
    length_3 = 0.5  # Length of the upper-right edge (from (1, 0.5) to (0.5, 0.5))
    length_4 = 0.5  # Length of the right edge (from (0.5, 0.5) to (0.5, 1))
    length_5 = 0.5  # Length of the upper edge (from (0.5, 1) to (0, 1))
    length_6 = 1  # Length of the left edge (from (0, 1) to (0, 0))

    # Calculate the number of points needed on each edge based on the target spacing
    num_points_1 = int(length_1 / target_spacing) + 1
    num_points_2 = int(length_2 / target_spacing) + 1
    num_points_3 = int(length_3 / target_spacing) + 1
    num_points_4 = int(length_4 / target_spacing) + 1
    num_points_5 = int(length_5 / target_spacing) + 1
    num_points_6 = int(length_6 / target_spacing) + 1

    # Generate points for each edge
    # Bottom edge (from (0, 0) to (1, 0))
    x1 = torch.linspace(0, 1, num_points_1)
    y1 = torch.zeros(num_points_1)
    
    # Right-bottom corner (from (1, 0) to (1, 0.5))
    x2 = torch.full((num_points_2,), 1)
    y2 = torch.linspace(0, 0.5, num_points_2)
    
    # Upper-right edge (from (1, 0.5) to (0.5, 0.5))
    x3 = torch.linspace(1, 0.5, num_points_3)
    y3 = torch.full((num_points_3,), 0.5)
    
    # Right edge (from (0.5, 0.5) to (0.5, 1))
    x4 = torch.full((num_points_4,), 0.5)
    y4 = torch.linspace(0.5, 1, num_points_4)
    
    # Upper edge (from (0.5, 1) to (0, 1))
    x5 = torch.linspace(0.5, 0, num_points_5)
    y5 = torch.ones(num_points_5)
    
    # Left edge (from (0, 1) to (0, 0))
    x6 = torch.zeros(num_points_6)
    y6 = torch.linspace(1, 0, num_points_6)

    # Concatenate all boundary points
    boundary_points = torch.cat([torch.stack((x1, y1), dim=1),
                                 torch.stack((x2, y2), dim=1),
                                 torch.stack((x3, y3), dim=1),
                                 torch.stack((x4, y4), dim=1),
                                 torch.stack((x5, y5), dim=1),
                                 torch.stack((x6, y6), dim=1)], dim=0)
    
    # Remove duplicate points at the junctions of edges
    boundary_points = torch.unique(boundary_points, dim=0)

    return boundary_points

# Generate boundary points for the L-shaped region
boundpoints = generate_l_shape_boundary_points(1/(num_points-1)).to(device)

# Function to generate inner points for the L-shaped region
def generate_l_shape_inner_points(num_points):
    """
    Generate uniformly distributed points inside the L-shaped region.
    The points are selected based on the condition that they lie inside the L-shape.
    """
    h = 2 / (num_points - 1)  # Grid spacing
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter out points in the upper-right quarter of the square
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    # Keep only the points inside the L-shaped region
    inside_l_shape = (grid_points[:,0] <= 0.5) | (grid_points[:,1] <= 0.5)
    grid_points = grid_points[inside_l_shape]
    
    # Convert numpy array to PyTorch tensor
    l_shape_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return l_shape_inner_points

# Generate inner points for the L-shaped region
trunk = generate_l_shape_inner_points(num_points).to(device)

# Define the DeepONet model with a linear branch
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network (linear layer without bias)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Define the trunk network with multiple layers
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
        
        # This tensor holds the weights for the dot product in the final layer
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the combined output using a dot product
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output
    
# Create the model and apply DataParallel for multi-GPU training
u = linear_branch_DeepONet(4*(num_points-1), 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap model in DataParallel

# Define the loss function (Mean Squared Error)
loss = torch.nn.MSELoss().to(device)

# Compute the gradient of the prediction with respect to the input x
def grad(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Loss function for interior points (LHS of the PDE)
def l_i(u, inputs1):
    """
    Compute the loss for interior points based on the Laplace equation.
    This loss term corresponds to the left-hand side of the PDE (the Laplacian of u).
    """
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    x = trunk_batch[:,0:1].requires_grad_(True)
    y = trunk_batch[:,1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    # Expand branch input and repeat for each point
    branch_batch = inputs1[:, :4*(num_points-1)].unsqueeze(1).repeat(1, 1976, 1).view(-1, 4*(num_points-1)).to(device)
    
    # Make predictions using the model
    upred_batch = u(branch_batch, trunk_batch)
    
    # Calculate loss for the PDE (here it's Laplace, expected result is zero)
    l = loss(upred_batch, torch.zeros_like(upred_batch))
    return l

# Loss function for boundary points (RHS of the PDE)
def l_b(u, inputs2):
    """
    Compute the boundary loss for points on the boundary of the region.
    This loss term ensures that the solution satisfies the boundary conditions.
    """
    num_rows = inputs2.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    x = trunk_batch[:,0:1].requires_grad_(True)
    y = trunk_batch[:,1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    # Expand branch input and repeat for each point
    branch_batch = inputs2[:, :4*(num_points-1)].unsqueeze(1).repeat(1, 1976, 1).view(-1, 4*(num_points-1)).to(device)
    
    # Make predictions using the model
    upred_batch = u(branch_batch, trunk_batch)
    
    # Calculate loss for boundary conditions (expected result is zero)
    l = loss(upred_batch, torch.zeros_like(upred_batch))
    return l

# File path for training data
file_path = f"data/PINNlaplace2DL_{num_functions, num_points}.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching the dataset
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the optimizer and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load model checkpoint if available
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

# Set number of training epochs
epoches = 200000
model_path = f"models/PINNlaplace2DL_{num_functions, num_points}.pth"

start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        loss_function = 0.9*l_i(u, inputs) + 0.1*l_b(u, inputs)  # Total loss is a weighted sum of PDE and boundary losses
        l = loss_function.item()
        
        # Save the model if the current loss is the minimum encountered so far
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Use .module for saving
            minloss = l
        
        optimizer.zero_grad()
        loss_function.backward()  # Backpropagate the gradients
        optimizer.step()  # Update model parameters

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

# Calculate and print the total training time
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")

