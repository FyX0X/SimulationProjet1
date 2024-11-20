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


def get_theta(x,y):
    return math.atan(-x/y)