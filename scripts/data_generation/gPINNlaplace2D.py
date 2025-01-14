import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps  # Import the function for generating boundary data

# Parameter settings
num_functions = 200  # Number of functions to generate
num_points = 51  # Number of grid points inside the domain

def generate_training_data(num_networks, num_points):
    """
    Generate training data for solving the Laplace equation.

    :param num_networks: Number of functions to generate
    :param num_points: Number of interior grid points
    :return: Normalized training data, shape (num_networks, data vector length)
    """
    # Generate boundary value samples, each containing the solution values at boundary points
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4*num_points-3, 0.1, num_networks))).view(num_networks, -1)
    all_data = []  # To store all sample data

    # Process and normalize each sample
    for i in range(num_networks):
        u_b = u_b_samples[i].view(1, -1)  # Boundary values for the current sample
        data_row = torch.cat((u_b,), 1)  # Combine boundary values into a single row vector
        max_value = torch.max(torch.abs(u_b))  # Compute the maximum absolute value
        normalized_vector = data_row / max_value  # Normalize the data
        all_data.append(normalized_vector.detach().numpy())  # Append the normalized result to the list

    return np.vstack(all_data)  # Return stacked array of all sample data

def save_training_data(data, file_path):
    """
    Save the generated training data to a file.

    :param data: Training data to save
    :param file_path: Target file path for saving
    """
    # Save data using space as the delimiter
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Record start time
start_time = time.time()

# Set the file path for saving
file_path = f"data/PINNlaplace2D_{num_functions,num_points}.txt"

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

# Record end time and calculate elapsed time
end_time = time.time()
time_taken = end_time - start_time

# Print saving information and elapsed time
print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
