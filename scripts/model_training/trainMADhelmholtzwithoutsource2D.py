import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000  # Number of functions in the dataset
num_points = 51  # Number of grid points in each dimension
k = 1  # Identifier for the model configuration
checkpoint_path = f"models/MAD{k}helmholtzwithoutsource2D_checkpoint_1_1.pth"  # Path to save/load model checkpoints

# Dataset class to load data from a file
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Parse the data into a list of lists
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset

    def __getitem__(self, index):
        # Return a tensor of data at the given index
        return torch.tensor(self.data[index], dtype=torch.float32)

# Clear the GPU cache to ensure no memory overload
torch.cuda.empty_cache()

# Determine if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate grid spacing based on the number of points
h = 1 / (num_points - 1)
# Generate grid points for a 2D domain
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert grid points to PyTorch tensor and move to the selected device (GPU or CPU)
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Define a linear branch network for DeepONet architecture
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch network (single linear layer with 1000 neurons)
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
            nn.Linear(common_features, 1000),  # Output layer of the trunk network
        )
        
        # Last layer's weights to perform dot product in the forward pass
        self.last_layer_weights = nn.Parameter(torch.randn(1000))

    def forward(self, branch_input, trunk_input):
        # Pass the inputs through their respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Combine the outputs using a weighted sum of the dot product between branch and trunk
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Initialize the model and wrap it with DataParallel for multi-GPU training (if available)
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap the model for multi-GPU support

# Define Mean Squared Error (MSE) loss function
loss = torch.nn.MSELoss().to(device)

# Compute the gradient of u with respect to x (first derivative)
def grad_x(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Compute the loss function for a given batch of inputs
def l_label(u, inputs1):
    num_rows = inputs1.size(0)  # Get the batch size
    trunk_batch = trunk.repeat(num_rows, 1)  # Repeat grid points for each sample in the batch
    utrue_batch = inputs1[:, 4 * (num_points - 1):].reshape(-1, 1)  # True solution (u)
    # Branch input (reshape and repeat to match the required dimensions)
    branch_batch = inputs1[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1))
    # Predict output using the model
    upred_batch = u(branch_batch, trunk_batch)

    # Compute the MSE loss between predicted and true values
    l = loss(upred_batch, utrue_batch)
    return l

# File path for the dataset
file_path = f"data/MAD{k}helmholtz_withoutsource_2D_{num_functions, num_points}_1.txt"
dataset = Dataset(file_path)  # Load the dataset

# Create a DataLoader for batching the dataset
batch_size = 100  # Batch size for training
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Adam optimizer and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if available to resume training
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Load model weights (note: use .module)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
    start_epoch = checkpoint['epoch']  # Start from saved epoch
    minloss = checkpoint['min_loss']  # Load minimum loss value
    loss0 = checkpoint['loss_records']  # Load loss history
    elapsed_time = checkpoint['elapsed_time']  # Load elapsed time
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with minloss {minloss}.")
else:
    start_epoch = 0  # Start fresh if no checkpoint is found
    minloss = 1  # Set initial loss to a high value
    loss0 = []  # Initialize loss history
    elapsed_time = 0  # Initialize elapsed time
    print("No checkpoint found, starting from scratch.")

epoches = 200000  # Total number of epochs for training
model_path = f"models/MAD{k}helmholtzwithoutsource2D_{num_functions, num_points}_1_1.pth"  # Model save path

# Track total training time
start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)  # Move batch to device (GPU/CPU)
        loss_function = l_label(u, inputs)  # Compute the loss for the batch
        l = loss_function.item()  # Get the loss value
        # Save model if the current loss is lower than the minimum loss
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save model weights (note: use .module)
            minloss = l  # Update minimum loss

        optimizer.zero_grad()  # Zero out gradients from the previous step
        loss_function.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights

    # Print progress for each epoch
    print(f"Epoch [{epoch + 1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())  # Record the loss

    # Optionally, you can add a learning rate scheduler step
    # scheduler.step(loss_function)

    # Save a checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        end_time = time.time()
        elapsed_time += end_time - start_time  # Update elapsed time
        start_time = end_time  # Reset the timer
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': u.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': minloss,
            'loss_records': loss0,
            'elapsed_time': elapsed_time
        }
        torch.save(checkpoint, checkpoint_path)  # Save the checkpoint
        print(f"Checkpoint saved at epoch {epoch + 1} with elapsed time {elapsed_time:.2f} seconds.")

# Print the total time taken for training
end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")
