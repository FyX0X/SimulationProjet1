import numpy as np
import math


def load_data(filename):
    time = []
    theta = []
    max_x = 0
    min_x = 0
    with open(filename, 'r') as file:
        for line in file:
            
            formated = line.strip().split()
            
            if formated == ["mass", "A"] or formated == ['t', 'x', 'y']:
                continue
            
            time.append(float(formated[0]))
            x = float(formated[1])
            y = float(formated[2])
            max_x = max(max_x, x)
            min_x = min(min_x, x)
            theta.append(get_theta(x,y))
    
    print(f"max x: {max_x}")
    print(f"min x: {min_x}")
    print(theta[0])
            
    return np.array(time), np.array(theta)


def load_data_with_cart_movement(filename):
    time = []
    theta = []
    cart_x = []
    max_x = 0
    min_x = 0
    with open(filename, 'r') as file:
        for line in file:

            formated = line.strip().split()

            if formated == ['#multi:'] or formated == ["cart", "mass", "A"] or formated == ['t', 'x', 'y', 'x', 'y']:
                continue

            time.append(float(formated[0]))
            x_cart = float(formated[1])
            cart_x.append(x_cart)
            y_cart = float(formated[2])
            x_mass = float(formated[3])
            y_mass = float(formated[4])
            delta_x = x_mass - x_cart
            delta_y = y_mass - y_cart
            max_x = max(max_x, delta_x)
            min_x = min(min_x, delta_y)
            theta.append(get_theta(delta_x, delta_y))

    print(f"max x: {max_x}")
    print(f"min x: {min_x}")
    print(theta[0])

    return np.array(time), np.array(theta), np.array(cart_x)


def get_theta(x,y):
    return math.atan(-x/y)
