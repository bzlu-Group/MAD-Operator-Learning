import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000  # Number of functions in the dataset
num_points = 51  # Grid size (number of points)
item = 1  # MAD1 or MAD2
checkpoint_path = f"models/MADlaplace2D{item}circ_checkpoint_1_1.pth"  # Path to save/load the model checkpoint

class Dataset(Dataset):
    def __init__(self, file_path):
        # Read the dataset from the given file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return a single sample as a tensor
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()  # Free any unused memory on the GPU

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_circle_inner_points(num_points):
    """
    Generates points uniformly distributed within the unit circle.
    Args:
        num_points (int): Number of points to generate.
    Returns:
        circle_inner_points (Tensor): A tensor of points inside the unit circle.
    """
    h = 2 / (num_points - 1)  # Grid step size
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    x_grid, y_grid = np.meshgrid(x, y)

    # Filter out points outside the unit circle
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside_circle = np.sum(grid_points**2, axis=1) <= 1
    num_inside_circle = np.sum(inside_circle)
    print(f"Number of points inside the unit circle: {num_inside_circle}")
    
    # Only retain points inside the unit circle
    grid_points = grid_points[inside_circle]  
    
    # Convert to PyTorch tensor
    circle_inner_points = torch.tensor(grid_points, dtype=torch.float32)
    return circle_inner_points

# Generate the grid points inside the unit circle
trunk = generate_circle_inner_points(num_points).to(device)

class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        """
        Initialize the DeepONet model with separate branch and trunk networks.
        Args:
            branch_features (int): Number of features for the branch input.
            trunk_features (int): Number of features for the trunk input.
            common_features (int): Number of common features shared between the branch and trunk networks.
        """
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
        """
        Forward pass of the DeepONet model.
        Args:
            branch_input (Tensor): The input to the branch network.
            trunk_input (Tensor): The input to the trunk network.
        Returns:
            Tensor: The combined output after passing through both networks.
        """
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Perform the dot product between branch and trunk outputs
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Create the model and wrap it with DataParallel for multi-GPU support
u = linear_branch_DeepONet(100, 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap model for multi-GPU usage

# Loss function: Mean Squared Error (MSE) Loss
loss = torch.nn.MSELoss().to(device)

def grad_x(u, x):
    """
    Compute the gradient of u with respect to x.
    Args:
        u (Tensor): The function whose gradient is to be computed.
        x (Tensor): The input variable with respect to which the gradient is computed.
    Returns:
        Tensor: The computed gradient.
    """
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

def l_label(u, inputs1):
    """
    Compute the loss function for training.
    Args:
        u (Model): The trained model.
        inputs1 (Tensor): The input tensor containing the branch and trunk information.
    Returns:
        Tensor: The computed loss value.
    """
    num_rows = inputs1.size(0)
    
    # Prepare the trunk and branch inputs for the model
    trunk_batch = trunk.repeat(num_rows, 1)
    utrue_batch = inputs1[:, 100:].reshape(-1, 1)  # True values for the function
    branch_batch = inputs1[:, 0:100].unsqueeze(1).repeat(1, 1957, 1).view(-1, 100)
    
    # Get the predicted values from the model
    upred_batch = u(branch_batch, trunk_batch)
    
    # Calculate the loss between predicted and true values
    l = loss(upred_batch, utrue_batch)
    return l

# Path to the dataset file
file_path = f"data/MADlaplace2D{item}circ_{num_functions, num_points}_1.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(u.parameters(), lr=0.0001)  # Adam optimizer
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)  # Learning rate scheduler

# Load the model checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Load weights-only checkpoint
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Load model state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state dict
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state dict
    start_epoch = checkpoint['epoch']  # Resume from last saved epoch
    minloss = checkpoint['min_loss']  # Minimum loss from previous training
    loss0 = checkpoint['loss_records']  # List of loss records
    elapsed_time = checkpoint['elapsed_time']  # Elapsed time
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with minloss {minloss}.")
else:
    start_epoch = 0
    minloss = 1
    loss0 = []
    elapsed_time = 0
    print("No checkpoint found, starting from scratch.")

epoches = 200000  # Total number of epochs for training
model_path = f"models/MADlaplace2D{item}circ_{num_functions, num_points}_1_1.pth"  # Model save path

# Start training
start_time = time.time()
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        
        # Calculate the loss for the current batch
        loss_function = l_label(u, inputs)
        l = loss_function.item()
        
        # Save the model if the current loss is lower than the minimum loss
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save the model's state_dict
            minloss = l
        
        optimizer.zero_grad()  # Clear previous gradients
        loss_function.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

    print(f"Epoch [{epoch+1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())  # Track the loss

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
