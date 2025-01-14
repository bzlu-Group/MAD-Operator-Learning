import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000  # Number of functions in the dataset
num_points = 51  # Number of grid points along each axis
checkpoint_path = f"models/MADpoisson2D_checkpoint.pth"  # Path for saving/loading model checkpoints

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

torch.cuda.empty_cache()  # Clear any existing CUDA cache

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)  # Grid spacing for the 2D grid
# Generate the 2D grid of points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert numpy array to PyTorch tensor and move to the selected device (CPU/GPU)
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Define the linear branch of DeepONet
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),  # Single linear layer for the branch
        )
        
        # Define the trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_features, common_features),  # Trunk network layers
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, 1000),  # Final output layer
        )
        
        # This tensor will hold the last layer's weights for the dot product
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Combine the outputs from branch and trunk networks by dot product with weights
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output
    
# Define the Poisson-specific DeepONet model with two branches
class possionONet(nn.Module):
    def __init__(self, branch1_features, branch2_features, trunk_features, common_features1, common_features2):
        super(possionONet, self).__init__()
        # Initialize the two branch networks with different input features
        self.net1 = linear_branch_DeepONet(branch1_features, trunk_features, common_features1)
        self.net2 = linear_branch_DeepONet(branch2_features, trunk_features, common_features2)

    def forward(self, branch1_input, branch2_input, trunk_input):
        # Pass inputs through both branch networks and the trunk network
        output1 = self.net1(branch1_input, trunk_input)
        output2 = self.net2(branch2_input, trunk_input)
        
        # Combine the outputs from both branches (addition)
        combined_output = output1 + output2
        return combined_output

# Create the model and wrap it with DataParallel for multi-GPU training
u = possionONet(4 * (num_points - 1), num_points * num_points, 2, 90, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap model for DataParallel (multi-GPU)

# Define the loss function
loss = torch.nn.MSELoss().to(device)

# Function to compute the gradient with respect to x (first derivative)
def grad_x(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Function to calculate the loss label
def l_label(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)  # Repeat the grid points for each batch
    utrue_batch = inputs1[:, num_points * num_points + 4 * (num_points - 1):].reshape(-1, 1).to(device)  # True u values
    branch1_batch = inputs1[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1)).to(device)
    branch2_batch = inputs1[:, 4 * (num_points - 1):num_points * num_points + 4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, num_points * num_points).to(device)
    upred_batch = u(branch1_batch, branch2_batch, trunk_batch)  # Model prediction

    # Compute and return the MSE loss
    l = loss(upred_batch, utrue_batch)
    return l

# File path to the dataset
file_path = f"data/MADpoisson2D_{num_functions, num_points}.txt"
dataset = Dataset(file_path)  # Load the dataset

# Create DataLoader for batching
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the optimizer (Adam) and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if available to resume training
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
    start_epoch = checkpoint['epoch']  # Resume from the saved epoch
    minloss = checkpoint['min_loss']  # Load the minimum loss recorded
    loss0 = checkpoint['loss_records']  # Load the loss records
    elapsed_time = checkpoint['elapsed_time']  # Load the elapsed time
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with minloss {minloss}.")
else:
    start_epoch = 0  # Start from scratch if no checkpoint is found
    minloss = 1  # Set initial loss to a high value
    loss0 = []  # Initialize loss records
    elapsed_time = 0  # Initialize elapsed time
    print("No checkpoint found, starting from scratch.")

epoches = 200000  # Total number of epochs for training
model_path = f"models/MADpoisson2D_{num_functions, num_points}.pth"  # Path to save the trained model

start_time = time.time()  # Start timer for training
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)  # Move the batch to the device (CPU/GPU)
        loss_function = l_label(u, inputs)  # Compute the loss
        l = loss_function.item()  # Get the loss value

        # Save model if current loss is lower than the minimum loss
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save model state
            minloss = l  # Update minimum loss

        optimizer.zero_grad()  # Zero the gradients
        loss_function.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

    # Print progress for each epoch
    print(f"Epoch [{epoch + 1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())  # Record the loss

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        end_time = time.time()
        elapsed_time += end_time - start_time  # Update elapsed time
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
        torch.save(checkpoint, checkpoint_path)  # Save checkpoint
        print(f"Checkpoint saved at epoch {epoch + 1} with elapsed time {elapsed_time:.2f} seconds.")

end_time = time.time()  # End timer
total_time = elapsed_time + (end_time - start_time)  # Total time taken for training
print(f"Total Time taken: {total_time:.2f} seconds.")

