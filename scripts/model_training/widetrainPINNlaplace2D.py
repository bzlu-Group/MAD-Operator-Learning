import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000
num_points = 51
checkpoint_path = f"models/widePINNlaplace2D_checkpoint.pth"

# Dataset class to load data from a file
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

# Clear any cached data in the GPU memory
torch.cuda.empty_cache()

# Check device availability (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)
# Generate grid points for the mesh
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert numpy array to PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Function to generate boundary points for a square grid
def generate_square_points(num_points):
    points_per_edge = num_points
    # Edge 1: From (0, 0) to (1, 0)
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge-1)), dim=1)
    # Edge 2: From (1, 0) to (1, 1)
    edge2 = torch.stack((torch.ones(points_per_edge-1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    # Edge 3: From (1, 1) to (0, 1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge-1)), dim=1)
    # Edge 4: From (0, 1) to (0, 0)
    edge4 = torch.stack((torch.zeros(points_per_edge-1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    # Concatenate all the edges and return the complete square boundary points
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

# Boundary points for the square grid
boundpoints = generate_square_points(num_points).to(device)

# Define the model architecture
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 2000, bias=False),
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
            nn.Linear(common_features, 2000),
        )
        
        # Define the last layer weights for the dot product operation
        self.last_layer_weights = nn.Parameter(torch.randn(2000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through the branch and trunk networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the combined output as the dot product of branch and trunk outputs
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Create the model and use DataParallel for multi-GPU support
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 220).to(device)
u = torch.nn.DataParallel(u)  # Wrap model with DataParallel

# Loss function (Mean Squared Error)
loss = torch.nn.MSELoss().to(device)

# Function to compute the gradient with respect to x and y
def grad(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Define the interior loss term
def l_i(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    x = trunk_batch[:, 0:1].requires_grad_(True)
    y = trunk_batch[:, 1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    branch_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1)).to(device)
    
    upred_batch = u(branch_batch, trunk_batch)
    # Compute second derivatives with respect to x and y
    upp = grad(grad(upred_batch, x), x) + grad(grad(upred_batch, y), y)

    l = loss(upp, torch.zeros_like(upp))  # Loss for the second derivatives
    return l

# Define the boundary loss term
def l_b(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = boundpoints.repeat(num_rows, 1)
    
    boundvalue = inputs1[:, :4 * (num_points - 1)].reshape(-1, 1).to(device)
    branch_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, 4 * (num_points - 1), 1).view(-1, 4 * (num_points - 1)).to(device)
    
    upred = u(branch_batch, trunk_batch)
    
    # Boundary loss: compare predicted and true boundary values
    l = loss(upred, boundvalue)
    return l

# File path to the dataset
file_path = f"data/PINNlaplace2D_{num_functions, num_points}.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if it exists, otherwise start fresh
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Load model state dict
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

# Total number of epochs
epoches = 200000
model_path = f"models/widePINNlaplace2D_{num_functions, num_points}.pth"

# Start the training loop
start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        # Total loss as a weighted sum of the interior and boundary losses
        loss_function = 0.9 * l_i(u, inputs) + 0.1 * l_b(u, inputs)
        l = loss_function.item()
        
        # Save model if new minimum loss is achieved
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save model state dict
            minloss = l
        
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

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

# Total time taken for training
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")

