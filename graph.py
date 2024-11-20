import numpy as np
from matplotlib import pyplot as plt
import load_tracker_data
import simulation


AUTO_CALIBRATE = True

_TIME_SHIFT = 0.065		# delta time between the data and the simulation, modify this value to shift simulation on the time axis
_THETA_SHIFT = 0        # only used if AUTO_CALIBRATE IS FALSE


def plot_data_and_simulation():
    plt.figure(1)
    plt.plot(sim_time, sim_theta , label="simulation")
    plt.plot(data_time, data_theta, label="data")
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.legend()
    plt.show()


def get_theta_0(theta):
    max_theta = find_max(theta)
    min_theta = find_max(theta*-1)

    average_theta = 0

    for index in max_theta:
        average_theta += theta[index]
    for index in min_theta:
        average_theta += theta[index]

    average_theta /= (len(max_theta) + len(min_theta))

    return average_theta


def get_delta_theta(theta_data, theta_sim):
    delta = get_theta_0(theta_data) - get_theta_0(theta_sim)
    return delta


def get_time_diffs(data_time, data_theta, sim_time, sim_theta):
    time_differences = []
    avg_time_diff = 0

    ### plot difference in maxima time for simulation synchronisation
    max_data = find_max(data_theta)
    max_sim = find_max(sim_theta)

    for i in zip(max_data, max_sim):
        time_diff = data_time[i[0]] - sim_time[i[1]]
        time_differences.append(time_diff)
        avg_time_diff += time_diff

    avg_time_diff /= len(time_differences)

    return time_differences, avg_time_diff


def get_theta_max_diffs(data_theta, sim_theta):
    theta_max_differences = []

    max_data = find_max(data_theta)
    max_sim = find_max(sim_theta)

    for i in zip(max_data, max_sim):
        theta_diff = data_theta[i[0]] - sim_theta[i[1]]
        theta_max_differences.append(theta_diff)

    return theta_max_differences


def plot_time_diffs(time_differences):
    plt.figure(2)
    print("test")
    plt.plot(list(range(len(time_differences))), time_differences)
    plt.xlabel("corresponding maxima number")
    plt.ylabel("time difference [s]")
    plt.show()


def plot_theta_max_diffs(theta_max_differences):
    plt.figure(3)
    plt.plot(list(range(len(theta_max_differences))), theta_max_differences)
    plt.xlabel("corresponding maxima number")
    plt.ylabel("theta difference [rad]")
    plt.show()



def find_max(array):
    max_lst = []
    for i in range(1, len(array)-1):
        if array[i] > array[i-1] and array[i] > array[i+1]:
            # array[i] is local max
            max_lst.append(i)
    return max_lst
        

if __name__ == '__main__':
    
    simulation.simulate("const", 0.85, 1.3)

    data_time, data_theta = load_tracker_data.load_data("data4.txt")

    if AUTO_CALIBRATE:
        time_differences, delta_time = get_time_diffs(data_time, data_theta, simulation.t, simulation.theta)
        delta_theta = get_delta_theta(data_theta, simulation.theta)
        print(f"base delta time = {delta_time}")
        print(f"base delta theta = {delta_theta}")
    else:
        delta_time = _TIME_SHIFT
        delta_theta = _THETA_SHIFT
    sim_time = simulation.t + delta_time
    sim_theta = simulation.theta + delta_theta

    plot_data_and_simulation()

    new_diff_time, new_delta_time = get_time_diffs(data_time, data_theta, sim_time, sim_theta)
    new_delta_theta = get_delta_theta(data_theta, sim_theta)
    print(f"new delta time: {new_delta_time}")
    print(f"new delta theta: {new_delta_theta}")
    #plot_time_diffs(new_diff_time)

    new_diff_theta = get_theta_max_diffs(data_theta, sim_theta)
    #plot_theta_max_diffs(new_diff_theta)
    