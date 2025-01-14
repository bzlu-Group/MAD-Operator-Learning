import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os

num_functions = 2000
num_points = 51
item = 1  # MAD1 or MAD2
checkpoint_path = f"models/deepMADlaplace2D{item}_checkpoint.pth"

# Dataset class to load data from file
class Dataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        self.data = [[float(val) for val in line.strip().split()] for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

torch.cuda.empty_cache()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h = 1 / (num_points - 1)
# Generate grid points
x = np.arange(0, 1 + h, h)
y = np.arange(0, 1 + h, h)
x_grid, y_grid = np.meshgrid(x, y)
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Convert numpy array to PyTorch tensor
trunk = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Define the neural network architecture
class linear_branch_DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, common_features):
        super(linear_branch_DeepONet, self).__init__()
        # Define the branch net (for branch input)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_features, 1000, bias=False),
        )
        
        # Define the trunk net (for trunk input)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
            nn.Tanh(),
            nn.Linear(common_features, common_features),
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
        # Pass inputs through respective networks
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Perform element-wise multiplication and sum the result
        combined_output = torch.sum(branch_output * trunk_output * self.last_layer_weights, dim=1, keepdim=True)
        return combined_output

# Create the model and use DataParallel for multi-GPU training
u = linear_branch_DeepONet(4 * (num_points - 1), 2, 110).to(device)
u = torch.nn.DataParallel(u)  # Wrap model in DataParallel

# Loss function (Mean Squared Error)
loss = torch.nn.MSELoss().to(device)

# Function to compute gradient with respect to x
def grad_x(u, x):
    grad_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, only_inputs=True)[0]
    return grad_x

# Compute the loss function
def l_label(u, inputs1):
    num_rows = inputs1.size(0)
    trunk_batch = trunk.repeat(num_rows, 1)
    utrue_batch = inputs1[:, 4 * (num_points - 1):].reshape(-1, 1)
    branch_batch = inputs1[:, 0:4 * (num_points - 1)].unsqueeze(1).repeat(1, num_points * num_points, 1).view(-1, 4 * (num_points - 1))
    upred_batch = u(branch_batch, trunk_batch)

    l = loss(upred_batch, utrue_batch)
    return l

# File path for loading data
file_path = f"data/MADlaplace2D{item}_{num_functions, num_points}.txt"
dataset = Dataset(file_path)

# Create DataLoader for batching data
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(u.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.95)

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    u.module.load_state_dict(checkpoint['model_state_dict'])  # Note: use .module to access model state
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

# Number of epochs for training
epoches = 200000
model_path = f"model/deepMADlaplace2D{item}_{num_functions, num_points}.pth"

start_time = time.time()
# Training loop
for epoch in range(start_epoch, epoches):
    for i, batch in enumerate(dataloader):
        inputs = batch.to(device)
        loss_function = l_label(u, inputs)
        l = loss_function.item()
        if l < minloss:
            torch.save(u.module.state_dict(), model_path)  # Save model using .module
            minloss = l
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}], Minimum Loss: {minloss}, Total Loss: {l}")
    loss0.append(loss_function.item())
    # scheduler.step(loss_function)

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

end_time = time.time()
total_time = elapsed_time + (end_time - start_time)
print(f"Total Time taken: {total_time:.2f} seconds.")
