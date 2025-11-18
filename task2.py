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

# Create 2D contour plots
fig2, ax2 = plt.subplots(1, 1)           # 2D axes (not projection='3d')
contour = ax2.contour(x, y, f, 100, cmap='magma')
fig2.colorbar(contour)

# GRADIENT DESCENT START
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
# GRADIENT DESCENT END

# NEWTON'S METHOD START
# Calculate Hessian of the Rosenbrock function at a given point
def rosenbrock_hess(x, y):
    d2fdx2 = 1200*x**2 - 400*y + 2
    d2fdy2 = 200
    d2fdxdy = -400*x
    return np.array([[d2fdx2, d2fdxdy],
                     [d2fdxdy, d2fdy2]])

# Implement Newton's method
def newtons_method(start, n_iterations):
    point = np.array(start) # Starting point (x, y)
    path = [point.copy()]   # To store the path taken
    for _ in range(n_iterations):
        grad = rosenbrock_grad(point[0], point[1])      # Compute gradient
        hess = rosenbrock_hess(point[0], point[1])      # Compute Hessian
        hess_inv = np.linalg.inv(hess)                  # Invert Hessian
        point -= hess_inv @ grad                        # Update point. @ is matrix multiplication
        path.append(point.copy())                       # Store the new point
    return np.array(path)
# NEWTON'S METHOD END

# GAUSS-NEWTON METHOD START
# Implement Gauss-Newton method
def gauss_newton_method(start, n_iterations):
    point = np.array(start) # Starting point (x, y)
    path = [point.copy()]   # To store the path taken
    for _ in range(n_iterations):
        grad = rosenbrock_grad(point[0], point[1])      # Compute gradient
        J = np.array([[-20*point[0], 10], [-1, 0]])     # Jacobian matrix. See Lecture 2 Slide 25
        hess_gn = np.linalg.inv(J.T @ J)                # Invert J^T * J
        point -= 0.5 * hess_gn @ grad                   # Update point. @ is matrix multiplication
        path.append(point.copy())                       # Store the new point
    return np.array(path)
# GAUSS-NEWTON METHOD END


# Parameters for gradient descent
start_point1 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
start_point2 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
start_point3 = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
learning_rate = 0.001
iterations = 20000

# Choose gradient descent, Newton's method, or Gauss-Newton method
# path1 = gradient_descent(start_point1, learning_rate, iterations)
# path2 = gradient_descent(start_point2, learning_rate, iterations)
# path3 = gradient_descent(start_point3, learning_rate, iterations)
path1 = newtons_method(start_point1, 20)
path2 = newtons_method(start_point2, 20)
path3 = newtons_method(start_point3, 20)
# path1 = gauss_newton_method(start_point1, 20)
# path2 = gauss_newton_method(start_point2, 20)
# path3 = gauss_newton_method(start_point3, 20)

# Plot the path on the above contour plot
ax2.plot(path1[:,0], path1[:,1], 'r.-', label='Newton Path 1')  # Plot the path in red
ax2.plot(path2[:,0], path2[:,1], 'c.-', label='Newton Path 2')  # Plot the path in cyan
ax2.plot(path3[:,0], path3[:,1], 'm.-', label='Newton Path 3')  # Plot the path in magenta
ax2.plot(path1[-1,0], path1[-1,1], 'rx', label='End Point 1')  # Ending point in red cross
ax2.plot(path2[-1,0], path2[-1,1], 'cx', label='End Point 2')  # Ending point in cyan cross
ax2.plot(path3[-1,0], path3[-1,1], 'mx', label='End Point 3')  # Ending point in magenta cross
ax2.plot(start_point1[0], start_point1[1], 'ro', label='Start Point 1')  # Starting point in red
ax2.plot(start_point2[0], start_point2[1], 'co', label='Start Point 2')  # Starting point in cyan
ax2.plot(start_point3[0], start_point3[1], 'mo', label='Start Point 3')  # Starting point in magenta

ax2.plot(1, 1, 'go', label='Analytical Minimum Point')  # Analytical minimum point in green dot
ax2.legend()

plt.show()

