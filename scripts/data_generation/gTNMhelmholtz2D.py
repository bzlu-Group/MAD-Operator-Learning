import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps
from Gauss import generate_smooth_random_field as gsrf
from traditional_methods import solve_helmholtz2D_direct as solve

# Parameters
num_functions = 20  # Number of functions to generate
num_points = 51  # Number of grid points for the computational domain
k = 100  
fine_factor = 1

def generate_training_data(num_functions, num_points):
    """
    Generate training data for the Helmholtz equation in 2D.

    The training data consists of:
    - Boundary values (u_b) generated using Gaussian processes.
    - Laplace values (laplace) generated using smooth random fields.
    - Solutions (u) computed using traditional numerical methods.

    The data is normalized for numerical stability.

    :param num_functions: Number of functions (training samples) to generate
    :param num_points: Number of grid points along each axis
    :return: Normalized training data as a Numpy array
    """
    # Batch generation of Laplace samples (f) and boundary values (u_b)
    laplace_samples = torch.tensor(np.array(gsrf(num_points, 5, num_functions))).reshape(num_functions, -1)
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4 * num_points - 3, 0.1, num_functions))).view(num_functions, -1)
    
    # Normalize and process each sample
    all_data = []
    for i in range(num_functions):
        # Extract the i-th sample for Laplace values and boundary conditions
        laplace = laplace_samples[i].view(1, -1)
        u_b = u_b_samples[i].view(1, -1)

        # Solve the Helmholtz equation to compute the solution (u)
        u = torch.tensor(solve(u_b, laplace, num_points, fine_factor, k)).view(1, -1)

        # Combine boundary values, Laplace values, and the solution into one row
        data_row = torch.cat((u_b, laplace, u), 1)

        # Normalize the data row by its maximum absolute value
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value

        # Append the normalized data row to the results
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    """
    Save the generated training data to a file.

    The data is saved in a space-separated format, where each row contains:
    - Boundary values (u_b)
    - Laplace values (laplace)
    - Solution values (u)

    :param data: Training data to save
    :param file_path: Path to save the training data
    """
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

# Main execution
start_time = time.time()
file_path = f"data/TNM{k}helmholtz2D_{num_functions,num_points}.txt"  

# Generate and save the training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
