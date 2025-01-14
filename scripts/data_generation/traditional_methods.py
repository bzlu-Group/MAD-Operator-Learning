import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata

# Solves the 1D Poisson equation using the finite difference method with 0 boundary conditions
# upp: the source term (right-hand side), N: number of grid points
def solve_poisson1D_dif(upp, N=100):
    h = 1.0 / N  # Grid spacing
    x = torch.linspace(0, 1, N+1)  # Generate grid points
    b = h**2 * upp  # Compute the right-hand side vector b

    # Create the tridiagonal matrix A corresponding to the discretized second-order differential operator
    A = torch.diag(torch.full((N-1,), -2.0)) + \
        torch.diag(torch.full((N-2,), 1.0), 1) + \
        torch.diag(torch.full((N-2,), 1.0), -1)

    # Solve the linear system Ax = b
    u = torch.zeros(N+1, 1)  # Initialize the solution vector
    b = b.view(-1, 1)  # Reshape to column vector
    u[1:N] = torch.linalg.solve(A, b[1:N])  # Solve for the interior points

    return u.view(-1, 1)  # Return the solution vector

# Set boundary conditions assuming boundary_values contains: bottom, right, top, and left (clockwise)
# u: solution matrix, boundary_values: vector of boundary values, num_points: number of grid points
def set_boundary_values(u, boundary_values, num_points=101):
    u[0, :] = boundary_values[0:num_points]  # Bottom boundary
    u[-1, :] = boundary_values[3*num_points-2:2*num_points-2:-1]  # Top boundary
    u[1:-1, 0] = boundary_values[4*num_points-4:3*num_points-3:-1]  # Left boundary
    u[:, -1] = boundary_values[num_points-1:2*num_points-1]  # Right boundary
    return u

# Generate the boundary points for a unit square with num_points on each edge
def generate_square_points(num_points):
    points_per_edge = num_points
    # Generate points for the four edges
    # First edge: from (0,0) to (1,0)
    edge1 = torch.stack((torch.linspace(0, 1, points_per_edge)[:-1], torch.zeros(points_per_edge-1)), dim=1)
    # Second edge: from (1,0) to (1,1)
    edge2 = torch.stack((torch.ones(points_per_edge-1), torch.linspace(0, 1, points_per_edge)[:-1]), dim=1)
    # Third edge: from (1,1) to (0,1)
    edge3 = torch.stack((torch.linspace(1, 0, points_per_edge)[:-1], torch.ones(points_per_edge-1)), dim=1)
    # Fourth edge: from (0,1) to (0,0)
    edge4 = torch.stack((torch.zeros(points_per_edge-1), torch.linspace(1, 0, points_per_edge)[:-1]), dim=1)
    complete_square = torch.cat((edge1, edge2, edge3, edge4), dim=0)
    
    return complete_square

# Generate grid points for a 2D grid with num_points in each direction
def generate_grid_points(num_points):
    x = np.linspace(0, 1, num_points)  # Generate x grid points
    y = np.linspace(0, 1, num_points)  # Generate y grid points
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")  # Create the meshgrid
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # Flatten and stack the grid points
    return grid_points, grid_x, grid_y

# Solve the 2D Laplace equation directly with given boundary conditions and grid points
# boundary_values: boundary condition values, num_points: number of grid points
def solve_laplace2D_direct(boundary_values, num_points=101):
    N = num_points - 2  # Number of interior grid points
    h = 1 / (num_points - 1)  # Grid spacing

    # Construct the sparse matrix for the Laplacian operator
    main_diag = -4 * np.ones(N * N)
    side_diag = np.ones(N * N - 1)
    side_diag[np.arange(1, N * N) % N == 0] = 0  # No adjacency for the last point in each row
    up_down_diag = np.ones(N * N - N)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    A = sp.diags(diagonals, [0, -1, 1, -N, N], format='csr')

    # Create the right-hand side vector and incorporate the boundary conditions
    b = np.zeros(N * N)
    boundary_values = boundary_values.reshape(-1)  # Flatten the boundary values array
    boundary_grid = np.zeros((num_points, num_points))
    boundary_grid[0, :] = boundary_values[0:num_points]  # Bottom boundary
    boundary_grid[-1, :] = boundary_values[3*num_points-2:2*num_points-2:-1]  # Top boundary
    boundary_grid[1:-1, 0] = boundary_values[4*num_points-4:3*num_points-3:-1]  # Left boundary
    boundary_grid[:, -1] = boundary_values[num_points-1:2*num_points-1]  # Right boundary

    b[:N] -= boundary_grid[0, 1:-1]   # Bottom boundary
    b[-N:] -= boundary_grid[-1, 1:-1]   # Top boundary
    b[::N] -= boundary_grid[1:-1, 0]   # Left boundary
    b[N-1::N] -= boundary_grid[1:-1, -1]  # Right boundary

    # Solve the linear system
    u_inner = sp.linalg.spsolve(A, b)
    u = np.zeros((num_points, num_points))
    u[1:-1, 1:-1] = u_inner.reshape((N, N))  # Fill in the interior solution

    # Add boundary values
    u[0, :] = boundary_grid[0, :]
    u[-1, :] = boundary_grid[-1, :]
    u[:, 0] = boundary_grid[:, 0]
    u[:, -1] = boundary_grid[:, -1]

    return u

def solve_laplace2D_direct_elaborate(boundary_values, num_points=101, fine_factor=1):
    N = num_points - 2  # Number of interior grid points
    h = 1 / (num_points - 1)  # Grid spacing

    # Refined grid settings
    fine_num_points = (num_points - 1) * fine_factor + 1  # Number of grid points in the fine grid
    fine_N = fine_num_points - 2  # Number of interior grid points in the fine grid
    fine_h = 1 / (fine_num_points - 1)  # Refined grid spacing

    # Construct the sparse matrix for the Laplacian operator on the refined grid
    main_diag = -4 * np.ones(fine_N * fine_N)  # Main diagonal (the center points)
    side_diag = np.ones(fine_N * fine_N - 1)  # Side diagonals (left-right neighbors)
    side_diag[np.arange(1, fine_N * fine_N) % fine_N == 0] = 0  # No adjacency for the last point in each row
    up_down_diag = np.ones(fine_N * fine_N - fine_N)  # Up and down diagonals (top-bottom neighbors)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    A = sp.diags(diagonals, [0, -1, 1, -fine_N, fine_N], format='csr')  # Sparse matrix for Laplacian

    # Initialize the right-hand side vector
    b = np.zeros(fine_N * fine_N)

    # Reshape boundary values and assign them to the boundary grid
    boundary_values = boundary_values.reshape(-1)  # Flatten the boundary values array
    boundary_grid = np.zeros((num_points, num_points))  # Initialize the grid for boundary conditions
    boundary_grid[0, :] = boundary_values[0:num_points]  # Bottom boundary

    # Define a function to reverse an array (avoid using negative step slicing)
    def reverse_array(a):
        return a[np.arange(len(a)-1, -1, -1)]

    # Top boundary
    tmp = boundary_values[2*num_points-2:3*num_points-2]
    boundary_grid[-1, :] = reverse_array(tmp)  # Reverse and assign to the top boundary

    # Left boundary
    tmp = boundary_values[3*num_points-2:4*num_points-4]
    boundary_grid[1:-1, 0] = reverse_array(tmp)  # Reverse and assign to the left boundary

    # Right boundary
    boundary_grid[:, -1] = boundary_values[num_points-1:2*num_points-1]  # Right boundary

    # Create grid points for interpolation
    x, y = np.meshgrid(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points))
    fine_x, fine_y = np.meshgrid(np.linspace(0, 1, fine_num_points), np.linspace(0, 1, fine_num_points))

    # Interpolate boundary values onto the fine grid
    points = np.vstack((x.flatten(), y.flatten())).T  # Flatten original grid points
    values = boundary_grid.flatten()  # Flatten boundary grid values
    fine_points = np.vstack((fine_x.flatten(), fine_y.flatten())).T  # Flatten fine grid points

    fine_boundary_grid = griddata(points, values, fine_points, method='linear')  # Perform linear interpolation
    fine_boundary_grid = fine_boundary_grid.reshape(fine_num_points, fine_num_points)  # Reshape to fine grid

    # Incorporate boundary values into the right-hand side vector
    b[:fine_N] -= fine_boundary_grid[0, 1:-1]   # Bottom boundary (excluding corners)
    b[-fine_N:] -= fine_boundary_grid[-1, 1:-1]   # Top boundary (excluding corners)
    b[::fine_N] -= fine_boundary_grid[1:-1, 0]   # Left boundary (excluding corners)
    b[fine_N-1::fine_N] -= fine_boundary_grid[1:-1, -1]  # Right boundary (excluding corners)

    # Solve the system of equations
    u_inner = spla.spsolve(A, b)  # Solve the linear system using sparse solver
    u_fine = np.zeros((fine_num_points, fine_num_points))  # Initialize the solution array for fine grid
    u_fine[1:-1, 1:-1] = u_inner.reshape((fine_N, fine_N))  # Fill the interior solution

    # Assign boundary values to the solution
    u_fine[0, :] = fine_boundary_grid[0, :]  # Bottom boundary
    u_fine[-1, :] = fine_boundary_grid[-1, :]  # Top boundary
    u_fine[:, 0] = fine_boundary_grid[:, 0]  # Left boundary
    u_fine[:, -1] = fine_boundary_grid[:, -1]  # Right boundary

    # Interpolate the solution back to the coarse grid
    u_coarse = griddata(fine_points, u_fine.flatten(), points, method='linear')  # Interpolation
    u_coarse = u_coarse.reshape(num_points, num_points)  # Reshape back to coarse grid

    return u_coarse  # Return the solution on the coarse grid

def solve_poisson2D_direct_elaborate(boundary_values, source_term, num_points=101, fine_factor=1):
    N = num_points - 2  # Number of interior grid points for the coarse grid
    h = 1 / (num_points - 1)  # Grid spacing for the coarse grid

    # Refined grid settings
    fine_num_points = (num_points - 1) * fine_factor + 1  # Number of grid points in the fine grid
    fine_N = fine_num_points - 2  # Number of interior grid points for the fine grid
    fine_h = 1 / (fine_num_points - 1)  # Refined grid spacing

    # Construct the sparse matrix for the Laplacian operator on the fine grid
    main_diag = -4 * np.ones(fine_N * fine_N)  # Main diagonal (center points)
    side_diag = np.ones(fine_N * fine_N - 1)  # Side diagonals (left-right neighbors)
    side_diag[np.arange(1, fine_N * fine_N) % fine_N == 0] = 0  # No adjacency for the last point in each row
    up_down_diag = np.ones(fine_N * fine_N - fine_N)  # Up and down diagonals (top-bottom neighbors)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    A = sp.diags(diagonals, [0, -1, 1, -fine_N, fine_N], format='csr')  # Sparse matrix for the Laplacian

    # Initialize the right-hand side vector
    b = np.zeros(fine_N * fine_N)

    # Reshape the boundary values and assign them to the boundary grid
    boundary_values = boundary_values.reshape(-1)  # Flatten the boundary values array
    boundary_grid = np.zeros((num_points, num_points))  # Initialize the grid for boundary conditions
    boundary_grid[0, :] = boundary_values[0:num_points]  # Bottom boundary

    # Define a function to reverse an array (avoid using negative step slicing)
    def reverse_array(a):
        return a[np.arange(len(a)-1, -1, -1)]

    # Top boundary
    tmp = boundary_values[2*num_points-2:3*num_points-2]
    boundary_grid[-1, :] = reverse_array(tmp)  # Reverse and assign to the top boundary

    # Left boundary
    tmp = boundary_values[3*num_points-2:4*num_points-4]
    boundary_grid[1:-1, 0] = reverse_array(tmp)  # Reverse and assign to the left boundary

    # Right boundary
    boundary_grid[:, -1] = boundary_values[num_points-1:2*num_points-1]  # Right boundary

    # Create grid points for interpolation
    x, y = np.meshgrid(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points))
    fine_x, fine_y = np.meshgrid(np.linspace(0, 1, fine_num_points), np.linspace(0, 1, fine_num_points))

    # Interpolate the boundary values onto the fine grid
    points = np.vstack((x.flatten(), y.flatten())).T  # Flatten original grid points
    values = boundary_grid.flatten()  # Flatten boundary grid values
    fine_points = np.vstack((fine_x.flatten(), fine_y.flatten())).T  # Flatten fine grid points

    fine_boundary_grid = griddata(points, values, fine_points, method='linear')  # Perform linear interpolation
    fine_boundary_grid = fine_boundary_grid.reshape(fine_num_points, fine_num_points)  # Reshape to fine grid

    # Incorporate boundary values into the right-hand side vector
    b[:fine_N] -= fine_boundary_grid[0, 1:-1]   # Bottom boundary (excluding corners)
    b[-fine_N:] -= fine_boundary_grid[-1, 1:-1]   # Top boundary (excluding corners)
    b[::fine_N] -= fine_boundary_grid[1:-1, 0]   # Left boundary (excluding corners)
    b[fine_N-1::fine_N] -= fine_boundary_grid[1:-1, -1]  # Right boundary (excluding corners)

    # Interpolate the source term onto the fine grid and add it to the right-hand side vector
    source_term_fine = griddata(points, source_term.flatten(), fine_points, method='linear')  # Interpolate source term
    source_term_fine = source_term_fine.reshape(fine_num_points, fine_num_points)  # Reshape to fine grid
    b += source_term_fine[1:-1, 1:-1].flatten() * fine_h**2  # Add the source term to the right-hand side

    # Solve the system of equations
    u_inner = spla.spsolve(A, b)  # Solve the linear system using sparse solver
    u_fine = np.zeros((fine_num_points, fine_num_points))  # Initialize the solution array for fine grid
    u_fine[1:-1, 1:-1] = u_inner.reshape((fine_N, fine_N))  # Fill the interior solution

    # Assign boundary values to the solution
    u_fine[0, :] = fine_boundary_grid[0, :]  # Bottom boundary
    u_fine[-1, :] = fine_boundary_grid[-1, :]  # Top boundary
    u_fine[:, 0] = fine_boundary_grid[:, 0]  # Left boundary
    u_fine[:, -1] = fine_boundary_grid[:, -1]  # Right boundary

    # Interpolate the solution back to the coarse grid
    u_coarse = griddata(fine_points, u_fine.flatten(), points, method='linear')  # Interpolation
    u_coarse = u_coarse.reshape(num_points, num_points)  # Reshape back to coarse grid

    return u_coarse  # Return the solution on the coarse grid

def solve_helmholtz2D_direct(boundary_values, source_term, num_points=101, fine_factor=1, lambd=1):
    """
    Solve the Helmholtz equation: Δu + λu = f, with boundary condition u = g
    Parameters:
    - boundary_values: Boundary values g
    - source_term: Source term f
    - num_points: Number of grid points for the coarse grid
    - fine_factor: Refinement factor for the fine grid
    - lambd: Coefficient λ in the Helmholtz equation (default is 1)
    """
    N = num_points - 2  # Number of interior grid points for the coarse grid
    h = 1 / (num_points - 1)  # Grid spacing for the coarse grid

    # Refined grid settings
    fine_num_points = (num_points - 1) * fine_factor + 1  # Number of grid points in the fine grid
    fine_N = fine_num_points - 2  # Number of interior grid points for the fine grid
    fine_h = 1 / (fine_num_points - 1)  # Refined grid spacing

    # Construct the sparse matrix for the Helmholtz operator (Laplacian + λ)
    main_diag = (-4 + lambd*h*h) * np.ones(fine_N * fine_N)  # Main diagonal: -4 + λ
    side_diag = np.ones(fine_N * fine_N - 1)  # Side diagonals (left-right neighbors)
    side_diag[np.arange(1, fine_N * fine_N) % fine_N == 0] = 0  # No adjacency for the last point in each row
    up_down_diag = np.ones(fine_N * fine_N - fine_N)  # Up and down diagonals (top-bottom neighbors)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    A = sp.diags(diagonals, [0, -1, 1, -fine_N, fine_N], format='csr')  # Sparse matrix for the Helmholtz operator

    # Initialize the right-hand side vector
    b = np.zeros(fine_N * fine_N)

    # Reshape the boundary values and assign them to the boundary grid
    boundary_values = boundary_values.reshape(-1)  # Flatten the boundary values array
    boundary_grid = np.zeros((num_points, num_points))  # Initialize the grid for boundary conditions
    boundary_grid[0, :] = boundary_values[0:num_points]  # Bottom boundary

    # Define a function to reverse an array (avoid using negative step slicing)
    def reverse_array(a):
        return a[np.arange(len(a)-1, -1, -1)]

    # Top boundary
    tmp = boundary_values[2*num_points-2:3*num_points-2]
    boundary_grid[-1, :] = reverse_array(tmp)  # Reverse and assign to the top boundary

    # Left boundary
    tmp = boundary_values[3*num_points-2:4*num_points-4]
    boundary_grid[1:-1, 0] = reverse_array(tmp)  # Reverse and assign to the left boundary

    # Right boundary
    boundary_grid[:, -1] = boundary_values[num_points-1:2*num_points-1]  # Right boundary

    # Create grid points for interpolation
    x, y = np.meshgrid(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points))
    fine_x, fine_y = np.meshgrid(np.linspace(0, 1, fine_num_points), np.linspace(0, 1, fine_num_points))

    # Interpolate the boundary values onto the fine grid
    points = np.vstack((x.flatten(), y.flatten())).T  # Flatten original grid points
    values = boundary_grid.flatten()  # Flatten boundary grid values
    fine_points = np.vstack((fine_x.flatten(), fine_y.flatten())).T  # Flatten fine grid points

    fine_boundary_grid = griddata(points, values, fine_points, method='linear')  # Perform linear interpolation
    fine_boundary_grid = fine_boundary_grid.reshape(fine_num_points, fine_num_points)  # Reshape to fine grid

    # Incorporate boundary values into the right-hand side vector
    b[:fine_N] -= fine_boundary_grid[0, 1:-1]   # Bottom boundary (excluding corners)
    b[-fine_N:] -= fine_boundary_grid[-1, 1:-1]   # Top boundary (excluding corners)
    b[::fine_N] -= fine_boundary_grid[1:-1, 0]   # Left boundary (excluding corners)
    b[fine_N-1::fine_N] -= fine_boundary_grid[1:-1, -1]  # Right boundary (excluding corners)

    # Interpolate the source term onto the fine grid and add it to the right-hand side vector
    source_term_fine = griddata(points, source_term.flatten(), fine_points, method='linear')  # Interpolate source term
    source_term_fine = source_term_fine.reshape(fine_num_points, fine_num_points)  # Reshape to fine grid
    b += source_term_fine[1:-1, 1:-1].flatten() * fine_h**2  # Add the source term to the right-hand side

    # Solve the system of equations
    u_inner = spla.spsolve(A, b)  # Solve the linear system using sparse solver
    u_fine = np.zeros((fine_num_points, fine_num_points))  # Initialize the solution array for fine grid
    u_fine[1:-1, 1:-1] = u_inner.reshape((fine_N, fine_N))  # Fill the interior solution

    # Assign boundary values to the solution
    u_fine[0, :] = fine_boundary_grid[0, :]  # Bottom boundary
    u_fine[-1, :] = fine_boundary_grid[-1, :]  # Top boundary
    u_fine[:, 0] = fine_boundary_grid[:, 0]  # Left boundary
    u_fine[:, -1] = fine_boundary_grid[:, -1]  # Right boundary

    # Interpolate the solution back to the coarse grid
    u_coarse = griddata(fine_points, u_fine.flatten(), points, method='linear')  # Interpolation
    u_coarse = u_coarse.reshape(num_points, num_points)  # Reshape back to coarse grid

    return u_coarse  # Return the solution on the coarse grid



