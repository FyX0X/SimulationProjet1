import numpy as np
from matplotlib import pyplot as plt
from simulation import Pendulum, SimulatedPendulum


def load_data(filename: str):
    """
    Load pendulum data from a text file and compute the angular position (theta).

    This function reads the data from the specified file, assuming the file contains
    time, x, and y values. It computes the angular position (theta) using the formula:
        theta = atan(-x/y)

    Args:
        filename (str): The name of the file containing the data. The file should have columns
                         representing time, x, and y values.

    Returns:
        tuple: A tuple containing:
            - time (np.ndarray): The time points from the file.
            - theta (np.ndarray): The computed angular position at each time point.
    """
    (time, x, y) = np.loadtxt(filename).T
    theta = np.atan(-x/y)

    return time, theta


def theta_graph(pendulums: list[Pendulum]):
    """
    Plot the angular position (theta) of multiple pendulums over time.

    This function plots the angular position (theta) of each pendulum in the given list
    over time. Each pendulum is plotted on the same graph, with the corresponding time on
    the x-axis and the angular position (theta) on the y-axis.

    Args:
        pendulums (list): A list of Pendulum objects whose angular position (theta) will be plotted.
    """
    plt.figure(0)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.theta, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("angular position [rad]")
    plt.legend()
    plt.show()


def energy_graph(pendulums: list[Pendulum]):
    """
    Plot the energy with respect to time for multiple pendulums.

    This function creates a plot of kinetic, potential and total energy versus time for each pendulum in the list.
    Each pendulum is represented by three separate curve in the graph.

    Args:
        pendulums (list): A list of Pendulum objects whose phase space (theta vs omega) will be plotted.
    """
    plt.figure(1)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.get_kinetic_energy(), label=pendulum.name + " - E_KIN")
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.get_potential_energy(), label=pendulum.name + " - E_POT")
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.get_total_energy(), label=pendulum.name + " - E_TOT")
    plt.xlabel("time [s]")
    plt.ylabel("energy [J]")
    plt.legend()
    plt.show()


def overall_graph(pendulums: list[Pendulum]):
    """
    Generate a set of subplots for various attributes of the pendulums over time.

    This function creates a figure with multiple subplots, displaying the angular position,
    angular velocity, and other relevant attributes of each pendulum in the provided list.

    Args:
        pendulums (list): A list of Pendulum objects to be visualized.
    """
    plt.figure(2)
    plt.subplot(3, 2, 1)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.theta, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("angular position [rad]")
    plt.legend()
    plt.subplot(3, 2, 3)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.omega, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("angular velocity [rad/s]")
    plt.legend()
    plt.subplot(3, 2, 5)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.alpha, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("angular acceleration [rad/s^2]")
    plt.legend()
    plt.subplot(3, 2, 2)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.x_c, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("cart position [m]")
    plt.legend()
    plt.subplot(3, 2, 4)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.v_c, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("cart velocity [m/s]")
    plt.legend()
    plt.subplot(3, 2, 6)
    for pendulum in pendulums:
        time = pendulum.time + pendulum.time_shift
        plt.plot(time, pendulum.a_c, label=pendulum.name)
    plt.xlabel("time [s]")
    plt.ylabel("cart acceleration [m/s^2]")
    plt.legend()
    plt.show()


def phase_graph(pendulums: list[Pendulum]):
    """
    Plot the phase space (angular position vs angular velocity) for multiple pendulums.

    This function creates a plot of angular position (theta) versus angular velocity (omega)
    for each pendulum in the list. Each pendulum is represented by a separate curve in the graph.

    Args:
        pendulums (list): A list of Pendulum objects whose phase space (theta vs omega) will be plotted.
    """
    plt.figure(3)
    for pendulum in pendulums:
        plt.plot(pendulum.theta, pendulum.omega, label=pendulum.name)
    plt.xlabel("angular position [rad]")
    plt.ylabel("angular velocity [rad/s]")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_time, data_theta = load_data("data4.txt")
    data = Pendulum("data", data_time)
    data.theta = data_theta
    data.derive_angular_state()
    data.time_shift = -0.04

    sim = SimulatedPendulum("simulation", data.theta[0], 0, 0.00015)
    sim.simulate()

    move_sim = SimulatedPendulum("move sim", np.pi/6, 0, 0.00015)
    move_sim.set_cart_profile_from_position("sinus", 1, 0.33)
    move_sim.simulate()

    #overall_graph([data, sim])
    theta_graph([data, sim])
    #energy_graph([data])
    #energy_graph([sim])
    #theta_graph([move_sim])
    #energy_graph([move_sim])
    #phase_graph([sim])





