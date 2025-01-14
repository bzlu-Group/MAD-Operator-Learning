import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps
from Gauss import generate_smooth_random_field as gsrf

# 参数设置
num_functions = 2000  # Number of random functions to generate
num_points = 51       # Number of grid points per dimension

def generate_training_data(num_functions, num_points):
    """
    Generate training data consisting of Laplace samples and boundary values.
    
    Args:
        num_functions: Number of random functions to generate.
        num_points: Number of grid points per dimension for Laplace samples.
    
    Returns:
        A numpy array of normalized training data.
    """
    # Generate random fields for Laplace solutions and boundary values in batches
    laplace_samples = torch.tensor(np.array(gsrf(num_points, 5, num_functions))).reshape(num_functions, -1)
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4 * num_points - 3, 0.1, num_functions))).view(num_functions, -1)
    
    # Normalize each sample and combine results into a single dataset
    all_data = []
    for i in range(num_functions):
        laplace = laplace_samples[i].view(1, -1)
        u_b = u_b_samples[i].view(1, -1)
        data_row = torch.cat((u_b, laplace), 1)
        max_value = torch.max(torch.abs(data_row))  # Normalize by the maximum absolute value
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())

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

file_path = f"data/PINNpoisson2D_{num_functions,num_points}.txt"  
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
