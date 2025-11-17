import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 2.01, 0.01)
y = np.arange(-2, 2.01, 0.01)
x , y = np.meshgrid(x, y)

f = 100*(y-x**2)**2 + (1-x)**2

# Create a 3D surface plot
# fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax1.plot_surface(x, y, f, cmap='viridis')
# fig1.colorbar(surf)

# plt.show()

# Create a 2D contour plot
fig2, ax2 = plt.subplots()           # 2D axes (not projection='3d')
contour = ax2.contour(x, y, f, 100, cmap='magma')
fig2.colorbar(contour)


# Calculate gradient of the Rosenbrock function at a given point
def rosenbrock_grad(x, y):
    dfdx = -400*x*(y - x**2) - 2*(1 - x)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])

# Implement gradient descent
def gradient_descent(start, alpha, n_iterations):
    point = np.array(start) # Starting point (x, y)
    path = [point.copy()]   # To store the path taken
    for _ in range(n_iterations):
        grad = rosenbrock_grad(point[0], point[1])  # Compute gradient
        point -= alpha * grad                       # alpha * grad is alpha * p
        path.append(point.copy())                   # Store the new point
    return np.array(path)

# Parameters for gradient descent
start_point = (-1.2, 1.0)
learning_rate = 0.001
iterations = 20000

path = gradient_descent(start_point, learning_rate, iterations)

# Plot the path on the above contour plot
ax2.plot(path[:,0], path[:,1], 'r.-')  # Plot the path in red
ax2.plot(start_point[0], start_point[1], 'bo')  # Starting point in blue
ax2.plot(1, 1, 'go')  # Analytical minimum point in green dot
ax2.plot(path[-1,0], path[-1,1], 'gx')  # Ending point in green cross
plt.legend(['Gradient Descent Path', 'Start Point', 'Analytical Minimum Point', 'End Point/Numerical Minimum Point'])
plt.show()