import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Plot of sin(x)")
plt.grid(True)

plt.tight_layout()
plt.show(block=True)