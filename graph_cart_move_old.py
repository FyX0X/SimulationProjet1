import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import load_tracker_data_old
import simulation_old

def smooth_data(data, window_size, poly_order):
    """
    Smooths the input data using the Savitzky-Golay filter.

    Args:
        data (numpy array): Input data to be smoothed.
        window_size (int): Size of the filter window (must be odd).
        poly_order (int): Polynomial order for fitting (must be less than window_size).

    Returns:
        numpy array: Smoothed data.
    """
    return savgol_filter(data, window_length=window_size, polyorder=poly_order)
def plot_data_and_simulation(time, s_theta, d_theta, x_cart, v_cart, a_cart, smooth_x):
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(time, s_theta, label="simulation")
    plt.plot(time, d_theta, label="data")
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(time, x_cart, label="data")
    plt.plot(time, smooth_x)
    plt.xlabel("time [s]")
    plt.ylabel("cart position [m]")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(time, v_cart, label="data")
    plt.xlabel("time [s]")
    plt.ylabel("cart velocity [m/s]")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(time, a_cart, label="data")
    plt.xlabel("time [s]")
    plt.ylabel("cart acceleration [m/s^2]")
    plt.legend()
    plt.show()


def truncate_sim(s_time, s_theta, d_time):
    index = 0
    max_data = d_time[-1]
    for i in range(len(s_time)):
        if s_time[i] > max_data:
            index = i
            break
    return s_time[:index], s_theta[:index]

def calc_cart_acc(x_cart, time_cart):
    v = np.empty_like(x_cart)
    a = np.empty_like(x_cart)

    for i in range(1, len(x_cart)):

        dt = time_cart[i] - time_cart[i-1]

        v[i] = (x_cart[i] - x_cart[i-1]) / dt
        a[i] = (v[i] - v[i-1]) / dt

    v[0] = v[1]
    a[0] = 0

    return a
        

if __name__ == '__main__':

    data_time, data_theta, data_x_cart = load_tracker_data_old.load_data_with_cart_movement("data_move.txt")
    interp_data_x_pos_func = interp1d(data_time, data_x_cart, kind='linear', fill_value="extrapolate")
    interp_data_x = interp_data_x_pos_func(simulation_old.t)
    # Parameters for the Savitzky-Golay filter
    window_size = 1  # Choose an odd number (larger values give more smoothing)
    poly_order = 0  # Typically 2 or 3 works well

    # Apply smoothing
    smoothed_x_cart = smooth_data(interp_data_x, window_size, poly_order)

    # Plot original and smoothed data for comparison

    plt.plot(data_time, data_x_cart, label="Original Data")
    plt.plot(simulation_old.t, smoothed_x_cart, label="Smoothed Data", linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Cart Position [m]")
    plt.legend()
    plt.show()

    # reset theta_0
    simulation_old.theta_0 = data_theta[0]
    simulation_old.x_c_0 = smoothed_x_cart[0]

    # calculate acceleration of cart BEFORE Interpolating
    acc = calc_cart_acc(smoothed_x_cart, simulation_old.t)
    # plt.plot(data_time, data_x_cart, label="pos")
    # plt.plot(data_time, acc, label="acceleration")
    # plt.show()

    # Interpolate experimental data to simulation time points
    interp_data_func = interp1d(data_time, data_theta, kind='linear', fill_value="extrapolate")
    interp_data_x_pos_func = interp1d(data_time, data_x_cart, kind='linear', fill_value="extrapolate")
    interp_smooth_x_pos_func = interp1d(simulation_old.t, smoothed_x_cart, kind='linear', fill_value="extrapolate")
    interp_data_x_acc_func = interp1d(simulation_old.t, acc, kind='linear', fill_value="extrapolate")

    x_pos_interpolated = interp_data_x_pos_func(simulation_old.t)       # temporary x pos but to long => has to be truncated
    smooth_x_pos_interpolated = interp_smooth_x_pos_func(simulation_old.t)       # temporary x pos but to long => has to be truncated

    acc_interp = interp_data_x_acc_func(simulation_old.t)

    simulation_old.simulate("data_acc", 0.8587, 1.2375, acc_interp)
    sim_time, sim_theta = truncate_sim(simulation_old.t, simulation_old.theta, data_time)

    theta_data_interp = interp_data_func(sim_time)  # interpolated data (theta data relative to sim time)

    mse = np.mean((sim_theta - theta_data_interp) ** 2)
    print("mse", mse)


    x_pos_interpolated = interp_data_x_pos_func(sim_time)
    plot_data_and_simulation(sim_time, sim_theta, theta_data_interp, x_pos_interpolated, simulation_old.v_c[:len(sim_time)], simulation_old.a_c[:len(sim_time)], smooth_x_pos_interpolated[:len(sim_time)])

    plt.plot(sim_time, simulation_old.x_c[:len(sim_time)], label="sim")
    plt.plot(data_time, data_x_cart, label="data")
    plt.show()

