import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps
from Gauss import generate_smooth_random_field as gsrf
from traditional_methods import solve_poisson2D_direct_elaborate as solve

# Parameters
num_functions = 200
num_points = 51

def generate_training_data(num_functions, num_points):
    # Generate laplace and u_b samples in batches to avoid loops
    laplace_samples = torch.tensor(np.array(gsrf(num_points, 5, num_functions))).reshape(num_functions, -1)
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4*num_points-3, 0.1, num_functions))).view(num_functions, -1)
    
    all_data = []
    for i in range(num_functions):
        laplace = laplace_samples[i].view(1, -1)
        u_b = u_b_samples[i].view(1, -1)
        u = torch.tensor(solve(u_b, laplace, num_points, 1)).view(1,-1)
        data_row = torch.cat((u_b, laplace, u), 1)
        max_value = torch.max(torch.abs(data_row))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy())

    return np.vstack(all_data)

def save_training_data(data, file_path):
    # Save data with space delimiter
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/TNMpoisson2D_{num_functions,num_points}.txt"  

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
