import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000
num_points = 51
checkpoint_path = f"models/PINNlaplace2D_checkpoint_1_1.pth"

class Dataset(Dataset):
    def __init__(self, file_path):
        # Read data from the provided file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Process each line and convert it to a list of floats
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return the data as a PyTorch tensor
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)
# Generate grid points for the domain
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert the grid points to a PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

def generate_square_points(num_points):
    # Generate boundary points for the square domain
    points_per_edge = num_points
    # Define the four edges of the square
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge-1)), dim=1)
    edge2 = torch.stack((torch.ones(points_per_edge-1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge-1)), dim=1)
    edge4 = torch.stack((torch.zeros(points_per_edge-1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    # Combine the edges into one tensor and return
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    return complete_square

# Generate boundary points and move them to the specified device (GPU or CPU)
boundpoints = generate_square_points(num_points).to(device)

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
        
        # A learnable tensor to hold the last layer's weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass inputs through the branch and trunk networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the final output by performing a dot product
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output
    
# Initialize the model and use DataParallel for multi-GPU support
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap the model in DataParallel

# Define the loss function
loss = torch.nn.MSELoss().to(device)

def grad(u, x):
    # Compute the gradient of u with respect to x
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

def l_i(u, inputs1):
    # Compute the interior loss (PDE residual)
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    x = trunk_batch[:, 0:1].requires_grad_(True)
    y = trunk_batch[:, 1:2].requires_grad_(True)
    trunk_batch = torch.cat((x, y), dim=1)

    # Branch input: Forcing term (f) is provided as input
    branch_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1)).to(device)
    
    # Predict the solution using the model
    upred_batch = u(branch_batch, trunk_batch)
    
    # Compute the second-order derivatives (Laplace operator)
    upp = grad(grad(upred_batch, x), x) + grad(grad(upred_batch, y), y)

    # Calculate the loss function for interior points
    l = loss(upp, torch.zeros_like(upp))
    return l

def l_b(u, inputs1):
    # Compute the boundary loss: apply boundary conditions
    num_rows = inputs1.size(0)
    trunk_batch = boundpoints.repeat(num_rows, 1)
    
    # Extract boundary values from the dataset
    boundvalue = inputs1[:, :4 * (num_points - 1)].reshape(-1, 1).to(device)
    branch_batch = inputs1[:, :4 * (num_points - 1)].unsqueeze(1).repeat(1, 4 * (num_points - 1), 1).view(-1, 4 * (num_points - 1)).to(device)
    
    # Predict the solution using the model
    upred = u(branch_batch, trunk_batch)

    # Calculate the boundary loss
    l = loss(upred, boundvalue)
    return l

# File path for the dataset
file_path = f"data/PINNlaplace2D_{num_functions, num_points}_1.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching the dataset
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize optimizer (Adam) and learning rate scheduler (ReduceLROnPlateau)
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if it exists, otherwise start from scratch
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

epoches = 200000
model_path = f"models/PINNlaplace2D_{num_functions, num_points}_1_1.pth"

start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        # Compute the total loss (weighted combination of interior and boundary losses)
        loss_function = 0.9 * l_i(u, inputs) + 0.1 * l_b(u, inputs)
        l = loss_function.item()
        
        # Save the model if the loss is lower than the previous minimum loss
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save model using .module for DataParallel
            minloss = l
        
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    # Print the current loss values
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

# Print the total time taken for training
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")
