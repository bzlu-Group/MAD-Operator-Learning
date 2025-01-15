import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps
from traditional_methods import solve_helmholtz2D_direct as solve

# Parameter settings
num_functions = 200
fine_factor = 20
low_res_points = 51    # Number of coarse grid points
high_res_points = fine_factor*(low_res_points-1)+1
k_value_square = 1


def generate_training_data(num_functions, high_res_points):
    # Generate high-resolution boundary values
    u_b_samples_high = torch.tensor(
        np.array(ggps(0, 1, 4*high_res_points - 3, 0.1, num_functions))
    ).view(num_functions, -1)
    
    all_data = []  # Store all data

    for i in range(num_functions):
        u_b_high = u_b_samples_high[i].view(1, -1)
        
        # Solve for high-resolution solution
        u_high = torch.tensor(
            solve(
                u_b_high.numpy(),
                np.zeros((high_res_points, high_res_points)),
                high_res_points,
                1,
                k_value_square
            )
        ).view(1, -1)
        
        u_high_grid = u_high.view(high_res_points, high_res_points)
        
        # Extract coarse grid data
        u_low_grid = u_high_grid[::fine_factor, ::fine_factor]  # Sample every nth point
        u_low = u_low_grid.reshape(1, -1)
        u_b_low = u_b_high[:, ::fine_factor]
        
        # Concatenate boundary values and solution, then normalize
        data_row = torch.cat((u_b_low, u_low), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())
    
    return np.vstack(all_data)

def save_training_data(data, file_path):
    # Save data using space as delimiter
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = (
    f"data/TNM{k_value_square}helmholtz_withoutsource_2D_{num_functions,low_res_points}.txt"
)

# Generate and save training data
training_data = generate_training_data(num_functions, high_res_points, low_res_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
