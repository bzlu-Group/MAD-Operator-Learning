import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps  # Import the function for generating boundary data

# Parameter settings
num_functions = 2000  # Number of functions to generate
num_points = 51  

def generate_training_data(num_functions, num_points):
    """
    Generate training data for the Laplace equation in a unit circle.

    :param num_functions: Number of functions to generate
    :param num_points: Number of points on the boundary of the unit circle
    :return: Normalized training data as a NumPy array
    """
    # Generate boundary values (u_b) for all functions
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 2*num_points-1, 0.1, num_functions))).view(num_functions, -1)
    
    # Normalize data for each function
    all_data = []
    for i in range(num_functions):
        u_b = u_b_samples[i].view(1, -1)  # Extract boundary values for the i-th function
        data_row = torch.cat((u_b,), 1)  # Combine boundary values into a single row
        max_value = torch.max(torch.abs(data_row))  # Compute the maximum absolute value
        normalized_vector = data_row / max_value  # Normalize the data
        all_data.append(normalized_vector.detach().numpy())  # Add normalized data to the list

    return np.vstack(all_data)  # Stack all rows into a single array

def save_training_data(data, file_path):
    """
    Save the generated training data to a file.

    :param data: Training data to save
    :param file_path: Path of the file to save the data
    """
    # Save the data using a space delimiter
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Start timing
start_time = time.time()

# File path for saving the training data
file_path = f"data/PINNlaplace2Dcirc_{num_functions,num_points}.txt"

# Generate and save the training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

# End timing and calculate elapsed time
end_time = time.time()
time_taken = end_time - start_time

# Print the results
print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
