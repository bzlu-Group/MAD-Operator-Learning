import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

# Number of functions and grid points
num_functions = 2000
num_points = 51
item = 1 # MAD1 or MAD2
checkpoint_path = f"models/MADlaplace2D{item}_checkpoint.pth"  # Path to save the model checkpoint

# Custom dataset class
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Read data from file and convert it into a list
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return the data at the specified index
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()  # Clear GPU cache

# Check if GPU is available, use GPU if possible, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)  # Grid spacing
# Generate grid points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)  # Create 2D grid
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T  # Flatten the grid
# Convert grid points to PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Define a DeepONet model with a linear branch network
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),  # Input: branch_features, output: 1000-dimensional
        )
        
        # Define the trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_features, common_features),  # Input: trunk_features, output: common_features
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, 1000),  # Output: 1000-dimensional
        )
        
        # Define a parameter for the last layer's weights for the dot product calculation
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Compute the weighted sum of the outputs of both networks
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Create the DeepONet model and wrap it with DataParallel for multi-GPU training
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap the model for multi-GPU support

# Define MSE loss function
loss = torch.nn.MSELoss().to(device)

# Function to compute the gradient of u with respect to x
def grad_x(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Function to compute the loss
def l_label(u, inputs1):
    num_rows = inputs1.size(0)  # Get the batch size
    trunk_batch = trunk.repeat(num_rows, 1)  # Repeat trunk grid points for each batch
    utrue_batch = inputs1[:, 4 * (num_points - 1):].reshape(-1, 1)  # Extract true solution
    branch_batch = inputs1[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1))  # Extract branch input
    upred_batch = u(branch_batch, trunk_batch)  # Get the predicted output from the model

    l = loss(upred_batch, utrue_batch)  # Calculate the loss
    return l

# File path to the dataset
file_path = f"data/MADlaplace2D{item}_{num_functions, num_points}.txt"
dataset = Dataset(file_path)  # Load the dataset

# Create DataLoader to load data in batches
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Adam optimizer
optimizer = optim.Adam(u.parameters(), lr=0.0001)
# Define the learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Load model state dict using .module for DataParallel
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    minloss = checkpoint['min_loss']
    loss0 = checkpoint['loss_records']
    elapsed_time = checkpoint['elapsed_time']
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with minloss {minloss}.")
else:
    start_epoch = 0
    minloss = 1  # Initialize minimum loss
    loss0 = []
    elapsed_time = 0  # Initialize elapsed time
    print("No checkpoint found, starting from scratch.")

epoches = 200000  # Set maximum number of training epochs
model_path = f"models/MADlaplace2D{item}_{num_functions, num_points}.pth"  # Path to save the model

# Record the training start time
start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)  # Move data to GPU
        loss_function = l_label(u, inputs)  # Compute the loss
        l = loss_function.item()  # Get the loss for the current batch
        if l < minloss:  # Save the model if current loss is less than minimum loss
            torch.save(u.module.state_dict(), model_path)  # Save the model state using .module for DataParallel
            minloss = l  # Update minimum loss
        optimizer.zero_grad()  # Clear gradients
        loss_function.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update the model parameters

    print(f"Epoch [{epoch+1}], Minimum Loss: {minloss}, Total Loss: {l}")  # Output loss for the current epoch
    loss0.append(loss_function.item())  # Record loss for the epoch
    # scheduler.step(loss_function)  # Optional, update learning rate based on loss

    if (epoch + 1) % 10 == 0:  # Save checkpoint every 10 epochs
        end_time = time.time()
        elapsed_time += end_time - start_time  # Accumulate elapsed time
        start_time = end_time
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': u.module.state_dict(),  # Save model state
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': minloss,  # Minimum loss encountered
            'loss_records': loss0,  # Record of loss over epochs
            'elapsed_time': elapsed_time  # Total elapsed time
        }
        torch.save(checkpoint, checkpoint_path)  # Save checkpoint
        print(f"Checkpoint saved at epoch {epoch + 1} with elapsed time {elapsed_time:.2f} seconds.")

# Record the training end time
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)  # Calculate total training time
print(f"Total Time taken: {total_time:.2f} seconds.")
