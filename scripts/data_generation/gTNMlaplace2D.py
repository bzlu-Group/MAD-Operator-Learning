import torch
import numpy as np
import time
from Gauss import generate_gps_circ as ggps
from traditional_methods import solve_laplace2D_direct_elaborate as solve

# Parameters
num_functions = 200
num_points = 51

def generate_training_data(num_functions, num_points):
    u_b_samples = torch.tensor(np.array(ggps(0, 1, 4*num_points-3, 0.1, num_functions))).view(num_functions, -1)
    all_data = []  # Stores all data

    for i in range(num_functions):
        u_b = u_b_samples[i].view(1, -1)
        u = torch.tensor(solve(u_b, num_points, 1)).view(1, -1)
        data_row = torch.cat((u_b, u), 1)
        max_value = torch.max(torch.abs(u_b))
        normalized_vector = data_row / max_value
        all_data.append(normalized_vector.detach().numpy()) 

    return np.vstack(all_data)

def save_training_data(data, file_path):
    # Save data with space delimiter
    np.savetxt(file_path, data, delimiter=" ", fmt='%f')

start_time = time.time()
file_path = f"data/TNMlaplace2D_{num_functions,num_points}.txt"  

# Generate and save training data
training_data = generate_training_data(num_functions, num_points)
save_training_data(training_data, file_path)

end_time = time.time()
time_taken = end_time - start_time

print(f"Training data saved to {file_path}.")
print(f"Time taken: {time_taken:.2f} seconds.")
