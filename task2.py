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
start_point = (-np.random.uniform(-2, 2), -np.random.uniform(-2, 2))
learning_rate = 0.001
iterations = 20000

# Choose gradient descent, Newton's method, or Gauss-Newton method
# path = gradient_descent(start_point, learning_rate, iterations)
# path = newtons_method(start_point, 20)
path = gauss_newton_method(start_point, 20)

# Plot the path on the above contour plot
ax2.plot(path[:,0], path[:,1], 'r.-')  # Plot the path in red
ax2.plot(start_point[0], start_point[1], 'bo')  # Starting point in blue
ax2.plot(1, 1, 'go')  # Analytical minimum point in green dot
ax2.plot(path[-1,0], path[-1,1], 'gx')  # Ending point in green cross
plt.legend(['Gradient Descent Path', 'Start Point', 'Analytical Minimum Point', 'End Point/Numerical Minimum Point'])
plt.show()

