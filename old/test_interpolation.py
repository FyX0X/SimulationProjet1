import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Example Data
t_sim = np.linspace(0, 10, 100)  # Simulation time points
theta_sim = np.sin(t_sim)  # Example simulation angular position (sinusoidal)

# Experimental data (irregular time points)
t_exp = np.sort(np.random.uniform(0, 10, 30))  # Experimental time points
theta_exp = np.sin(t_exp) + 0.1 * np.random.randn(len(t_exp))  # Noisy experimental data

# Interpolate experimental data to simulation time points
interp_func = interp1d(t_exp, theta_exp, kind='linear', fill_value="extrapolate")
theta_exp_interp = interp_func(t_sim)

# Compute fitness metric (e.g., Mean Squared Error)
mse = np.mean((theta_sim - theta_exp_interp)**2)

# Plot for visualization
plt.figure(figsize=(10, 5))
plt.plot(t_sim, theta_sim, label="Simulation", linewidth=2)
plt.scatter(t_exp, theta_exp, color="red", label="Experimental (raw)")
plt.plot(t_sim, theta_exp_interp, '--', label="Experimental (interpolated)", alpha=0.7)
plt.legend()
plt.title(f"Fitness (MSE): {mse:.4f}")
plt.xlabel("Time")
plt.ylabel("Theta")
plt.grid()
plt.show()

print(f"Mean Squared Error (MSE): {mse:.4f}")
