import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = np.arange(-2, 2.01, 0.01)
y = np.arange(-2, 2.01, 0.01)
x , y = np.meshgrid(x, y)

f = 100*(y-x**2)**2 + (1-x)**2

surf = ax.plot_surface(x, y, f, cmap='viridis')
fig.colorbar(surf)

plt.show()