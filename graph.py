import numpy as np
from matplotlib import pyplot as plt
import load_tracker_data
import simulation

TIME_SHIFT = 0.065		# delta time between the data and the simulation, modify this value to shift simulation on the time axis

def plot_data_and_simulation():
    plt.figure(1)
    plt.plot(sim_time, sim_theta , label="simulation")
    plt.plot(data_time, data_theta, label="data")
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.legend()
    plt.show()

def plot_time_diff():
    difference = []
    for x in zip(max_data, max_sim):
        difference.append(x[0] - x[1])
    plt.figure(2)
    plt.title("time difference between corresponding maximas in data and simulation")
    plt.plot(list(range(len(difference))), difference)
    plt.xlabel("corresponding maxima number")
    plt.ylabel("time difference [s]")
    plt.show()


def find_max(array, time):
    max_lst = []
    for i in range(1, len(array)-1):
        if array[i] > array[i-1] and array[i] > array[i+1]:
            # array[i] is local max
            max_lst.append(time[i])
    return max_lst
        


if __name__ == '__main__':
    
    simulation.simulate("const")
    
    sim_time = simulation.t + TIME_SHIFT
    sim_theta = simulation.theta
    
    data_time, data_theta = load_tracker_data.load_data("data4.txt")

    
    plot_data_and_simulation()
    
    
    ### plot difference in maxima time for simulation synchronisation
    max_data = find_max(data_theta, data_time)
    max_sim = find_max(sim_theta, sim_time)
    plot_time_diff()

    