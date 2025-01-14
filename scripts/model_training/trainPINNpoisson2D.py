import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

# Number of functions and points
num_functions = 2000
num_points = 51
checkpoint_path = f"models/PINNpoisson2D_checkpoint.pth"

# Dataset class to load data from file
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Parse the file and store data as a list of lists
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

# Clear GPU memory
torch.cuda.empty_cache()

# Determine device availability (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)
# Generate grid points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert numpy array to PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Function to generate points on the square boundary
def generate_square_points(num_points):
    points_per_edge = num_points
    # Define edges of the square
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge - 1)), dim=1)
    edge2 = torch.stack((torch.ones(points_per_edge - 1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge - 1)), dim=1)
    edge4 = torch.stack((torch.zeros(points_per_edge - 1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    # Combine the four edges and return the boundary points
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

# Generate boundary points for the square
boundpoints = generate_square_points(num_points).to(device)

# Define the linear branch network of DeepONet
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Branch network (single layer with 1000 units)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Trunk network (multi-layer with Tanh activation functions)
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
        
        # Weights for the last layer (used in dot product)
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Forward pass through branch and trunk networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the final output as the dot product of branch and trunk outputs
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Define PoissonONet model, combining two branch networks
class possionONet(nn.Module):
    def __init__(self, branch1_features, branch2_features, trunk_features, common_features1, common_features2):
        super(possionONet, self).__init__()
        # Create two linear branch networks
        self.net1 = linear_branch_DeepONet(branch1_features, trunk_features, common_features1)
        self.net2 = linear_branch_DeepONet(branch2_features, trunk_features, common_features2)

    def forward(self, branch1_input, branch2_input, trunk_input):
        # Forward pass through both branch networks
        output1 = self.net1(branch1_input, trunk_input)
        output2 = self.net2(branch2_input, trunk_input)
        
        # Combine the outputs of the two branch networks
        combined_output = output1 + output2
        return combined_output

# Create the model and wrap it with DataParallel
u = possionONet(4 * (num_points - 1), num_points * num_points, 2, 90, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap the model for parallel processing on multiple GPUs

# Define the loss function (Mean Squared Error)
loss = torch.nn.MSELoss().to(device)

# Function to compute the gradient
def grad(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Loss for interior points (Laplace equation)
def l_i(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    x = trunk_batch[:, 0:1].requires_grad_(True)
    y = trunk_batch[:, 1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    laplace_batch = inputs1[:, 4 * (num_points - 1):].reshape(-1, 1).to(device)
    branch1_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1)).to(device)
    branch2_batch = inputs1[:, 4 * (num_points - 1):].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, num_points * num_points).to(device)
    
    # Get the prediction from the model
    upred_batch = u(branch1_batch, branch2_batch, trunk_batch)
    # Compute second derivatives (Laplace operator)
    upp = grad(grad(upred_batch, x), x) + grad(grad(upred_batch, y), y)

    # Calculate the loss
    l = loss(upp, laplace_batch)
    return l

# Loss for boundary points
def l_b(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = boundpoints.repeat(num_rows, 1)
    
    # Extract boundary values
    boundvalue = inputs1[:, :4 * (num_points - 1)].reshape(-1, 1).to(device)
    branch1_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, 4 * (num_points - 1), 1).view(-1, 4 * (num_points - 1)).to(device)
    branch2_batch = inputs1[:, 4 * (num_points - 1):].unsqueeze(1).repeat(1, 4 * (num_points - 1), 1).view(-1, num_points * num_points).to(device)
    
    # Get the model prediction for boundary points
    upred = u(branch1_batch, branch2_batch, trunk_batch)

    # Calculate the loss for boundary points
    l = loss(upred, boundvalue)
    return l

# File path for dataset
file_path = f"data/PINNpoisson2D_{num_functions, num_points}.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching the data
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the optimizer (Adam) and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Use .module when loading
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

# Set the number of epochs for training
epoches = 200000
model_path = f"models/PINNpoisson2D_{num_functions, num_points}.pth"

# Training loop
start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        # Compute the total loss as a weighted sum of interior and boundary losses
        loss_function = 0.9 * l_i(u, inputs) + 0.1 * l_b(u, inputs)
        l = loss_function.item()
        # Save the model if the loss is minimal
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save the model using .module
            minloss = l
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())

    # Periodically save checkpoints
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

# Calculate total time taken
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")
