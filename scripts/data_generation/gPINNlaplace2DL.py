import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps

# Parameters
num_functions = 2000  # Number of functions to generate
num_points = 51  # Number of points for the grid

def generate_training_data(num_functions, num_points):
    """
    Generate training data for the PINN model using the L-shaped region.

    The training data consists of boundary values (u_b) sampled along the 
    boundary of the L-shaped region. The values are normalized to ensure
    numerical stability during model training.

    :param num_functions: Number of functions to generate
    :param num_points: Number of points along each axis of the grid
    :return: Numpy array of normalized training data
    """
    # Batch generate boundary values (u_b) using Gaussian process samples
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4 * num_points - 3, 0.1, num_functions))).view(num_functions, -1)
    
    # Normalize each sample and store the results
    all_data = []
    for i in range(num_functions):
        u_b = u_b_samples[i].view(1, -1)  # Reshape boundary values
        data_row = torch.cat((u_b,), 1)  # Concatenate data (currently only u_b)
        max_value = torch.max(torch.abs(data_row))  # Compute maximum absolute value
        normalized_vector = data_row / max_value  # Normalize the data
        all_data.append(normalized_vector.detach().numpy())  # Append to the result list

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """
    Save the generated training data to a file.

    The data is saved in a space-separated format for compatibility with
    common data processing pipelines.

    :param data: Training data to save
    :param file_path: Path to save the training data
    """
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Main execution
start_time = time.time()
file_path = f"data/PINNlaplace2DL_{num_functions,num_points}.txt"

# Generate and save the training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
