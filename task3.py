import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Minimise using Nelder-Mead Simplex algorithm
def rosenbrock_func(point):
    x, y = point
    return 100*(y - x**2)**2 + (1 - x)**2

def nelder_mead(start, tol, n_iterations):
    result = optimize.minimize(rosenbrock_func, start, method='Nelder-Mead', tol=tol, options={'maxiter': n_iterations, 'disp': False, 'return_all': True})
    path = result.allvecs  # Store the path taken
    return np.array(path)

x = np.arange(-2, 2.01, 0.01)
y = np.arange(-2, 2.01, 0.01)
x , y = np.meshgrid(x, y)
f = rosenbrock_func((x, y))

start_point1 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
start_point2 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
start_point3 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
iterations = 500
tolerance = 1e-6

# Create 2D contour plots
fig, ax = plt.subplots(1, 1)           # 2D axes (not projection='3d')
contour = ax.contour(x, y, f, 100, cmap='magma')
fig.colorbar(contour)


# NELDER-MEAD SIMPLEX START
# Perform Nelder-Mead optimization
path1 = nelder_mead(start_point1, tolerance, iterations)
path2 = nelder_mead(start_point2, tolerance, iterations)
path3 = nelder_mead(start_point3, tolerance, iterations)
# NELDER-MEAD SIMPLEX END

# Plot the paths on the contour plots
ax.plot(path1[:,0], path1[:,1], 'r.-', label='Nelder Mead Path 1')  # Plot the path in red
ax.plot(path2[:,0], path2[:,1], 'c.-', label='Nelder Mead Path 2')  # Plot the path in cyan
ax.plot(path3[:,0], path3[:,1], 'm.-', label='Nelder Mead Path 3')  # Plot the path in magenta
ax.plot(path1[-1,0], path1[-1,1], 'rx', label='End Point 1')  # Ending point in red cross
ax.plot(path2[-1,0], path2[-1,1], 'cx', label='End Point 2')  # Ending point in cyan cross
ax.plot(path3[-1,0], path3[-1,1], 'mx', label='End Point 3')  # Ending point in magenta cross
ax.plot(start_point1[0], start_point1[1], 'ro', label='Start Point 1')  # Starting point in red
ax.plot(start_point2[0], start_point2[1], 'co', label='Start Point 2')  # Starting point in cyan
ax.plot(start_point3[0], start_point3[1], 'mo', label='Start Point 3')  # Starting point in magenta

ax.plot(1, 1, 'go', label='Analytical Minimum Point')  # Analytical minimum point in green dot
ax.legend()

plt.show()