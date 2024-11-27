import numpy as np
from scipy.interpolate import interp1d
from load_tracker_data_old import load_data
import simulation_old
import graph_old
from matplotlib import pyplot as plt


def truncate_sim(sim_time, sim_theta, data_time):
    index = 0
    max_data = data_time[-1]
    for i in range(len(sim_time)):
        if sim_time[i] > max_data:
            index = i
            break
    return sim_time[:index], sim_theta[:index]


# Experimental data (irregular time points)
data_time, data_theta = load_data("data4.txt")

# Interpolate experimental data to simulation time points
interp_data_func = interp1d(data_time, data_theta, kind='linear', fill_value="extrapolate")


powers = np.arange(1.235, 1.240, 0.0001)
coefs = np.arange(0.845, 0.865, 0.0001)
selected_coef = np.empty_like(powers)
fitness = np.empty_like(powers)         # fitness is calculated as Mean Square Error

best_index = None
best_fitness = None
print("begin")
# iterate through all powers to find the best one
for i in range(len(powers)):
    print(f"{100*i/len(powers)}%")
    power = powers[i]

    best_coef = None
    best_fitness_current = None
    # iterate through all friction coeffs to find the best one for each power
    for coef in coefs:

        # Simulation Data
        simulation_old.simulate("const", coef, power)

        time_differences, delta_time = graph_old.get_time_diffs(data_time, data_theta, simulation_old.t, simulation_old.theta)
        delta_theta = graph_old.get_delta_theta(data_theta, simulation_old.theta)

        sim_time, sim_theta = truncate_sim(simulation_old.t + delta_time, simulation_old.theta, data_time)
        sim_theta += delta_theta
        theta_data_interp = interp_data_func(sim_time)      # interpolated data

        mse = np.mean((sim_theta - theta_data_interp) ** 2)

        """
        plt.plot(sim_time, sim_theta, label="sim")
        plt.plot(simulation_old.t, simulation_old.theta, label="not calibrated")
        plt.plot(data_time, data_theta, label="data")
        plt.legend()
        plt.show()"""

        if best_fitness_current is None:
            best_fitness_current = mse
            best_coef = coef
        if mse < best_fitness_current:
            best_coef = coef
            best_fitness_current = mse

    selected_coef[i] = best_coef
    fitness[i] = best_fitness_current
    if best_fitness is None:
        best_fitness = best_fitness_current
        best_index = i
    if best_fitness_current < best_fitness:
        best_fitness = best_fitness_current
        best_index = i

print(f"index= {best_index}")
print(f"best fitness: {min(fitness)}")
print(f"best power: {powers[best_index]}")
print(f"best coef: {selected_coef[best_index]}")
print(f"coefs: {selected_coef}")
print(f"fitness: {fitness}")


