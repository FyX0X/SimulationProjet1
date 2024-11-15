This Projects simulates a pendulum on a moving cart.

The file load_tracker_data.txt is used to load the x and y coordinate of the pendulum at any moment using a video and the Tracker program.
simulation.py does the simulation of the pendulum and can be used on its own or as a module.
graph.py is a python program that graphs both the data from data4.txt (using load_tracker_data.py) and the simulation (using simulation.py) in order to synchronize our simulation and change its parameter to be the closest possible with the experimental data.
graph.py also plots a graph of the time difference of each local maxima is the data to match the data as closely as possible.
