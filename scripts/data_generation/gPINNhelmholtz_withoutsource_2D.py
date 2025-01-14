import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps

# 参数设置
num_functions = 2000  # Number of random functions to generate
num_points = 51       # Number of grid points per dimension
k = 1                 # Equation: △u+ku=0

def generate_training_data(num_functions, num_points):
    """
    Generate training data consisting of boundary values only (without source term).

    Args:
        num_functions: Number of random boundary functions to generate.
        num_points: Number of grid points per dimension.

    Returns:
        A numpy array containing normalized training data.
    """
    # Generate boundary values for all functions in a batch
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4 * num_points - 3, 0.1, num_functions))).view(num_functions, -1)
    
    all_data = []
    for i in range(num_functions):
        u_b = u_b_samples[i].view(1, -1)  # Extract boundary values for the i-th function
        data_row = torch.cat((u_b,), 1)  # Combine data into a single row (only boundary values here)
        max_value = torch.max(torch.abs(u_b))  # Compute max absolute value for normalization
        normalized_vector = data_row / max_value  # Normalize the boundary values
        all_data.append(normalized_vector.detach().numpy())  # Append normalized data

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """
    Save the training data to a text file with space-separated values.

    Args:
        data: The training data to save.
        file_path: Path to the output file.
    """
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Main script for generating and saving training data
start_time = time.time()

# File path for storing the generated dataset
file_path = f"data/PINN{k}helmholtz_withoutsource_2D_{num_functions,num_points}.txt"

# Generate and save the training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
