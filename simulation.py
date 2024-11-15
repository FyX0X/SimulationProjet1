import math
import matplotlib.pyplot as plt
import numpy as np

### Constantes

g = 9.81         # gravitation [m/s**2]

### Paramètres du système

l = 0.255         # longueur du pendule [m]
m = 0.01          # masse du prehenseur [kg]
b = 0.00015         # coefficient de frottement [kg*m^2/s]


### Paramètres de la simulation

step = 0.001                       # pas (dt) [s]
end = 30                           # durée [s]
theta_0 = 0.5555125041788009       # position angulaire initiale [rad]				(from experimental data)
theta_dot_0 = 0                    # vitesse_angulaire initiale [rad/s]
x_c_0 = 0                          # position du cart initiale [m]
v_c_0 = 0.0                        # vitesse du cart initiale initiale [m/s]



### mouvement chariot

USE_POS = True			# True: cart movement from position, False, from acceleration

### Fonction de déplacement du Cart

def get_cart_motion(time, mode, pulsation=1, amplitude=1):
    match mode:
        case "const":
            return 0
        case "sinus":
            return amplitude * math.sin(pulsation * time)
        case "square":
            if USE_POS:
                raise Exception("Square Function is not continuous, therefore not allowed to use for Position")
            s = math.sin(pulsation*time)
            if s != 0:
                return amplitude * abs(s)/s
            return 0
        case "triangle":
            y = pulsation*time/math.pi
            return amplitude * (abs(y-int(y)-0.5) * 4 - 1)


t = np.arange(0, end, step)
theta = np.empty_like(t)          
theta_dot = np.empty_like(t)
theta_dot_dot = np.empty_like(t)

x_c = np.empty_like(t)
v_c = np.empty_like(t)
a_c = np.empty_like(t)

e_pot = np.empty_like(t)
e_kin = np.empty_like(t)
e_tot = np.empty_like(t)

    
def simulation(motion_type: str):
    """
    pre: motion_type: string, which formula to use with the simulation for the cart movement ("const", "sinus", "triangle", "square")
    post: exécute une simulation jusqu'à t=end par pas de dt=step.
          Remplit les listes theta, theta_dot, theta_dot_dot
          avec les positions, vitesses et accélérations angulaire du pendule.
    """
    # conditions initiales
    theta[0] = theta_0
    theta_dot[0] = theta_dot_0
    
    x_c[0] = x_c_0
    v_c[0] = v_c_0
    
    for i in range(len(t)-1):

        dt = step
        
        # calcule de la position du cart
        if USE_POS:
            x_c[i+1] = get_cart_motion(t[i], motion_type)
            v_c[i+1] = (x_c[i+1]-x_c[i])/dt
            a_c[i+1] = (v_c[i+1]-v_c[i])/dt
        else:
            a_c[i] = get_cart_motion(t[i], motion_type)
            v_c[i+1] = v_c[i] + a_c[i] * dt
            x_c[i+1] = x_c[i] + v_c[i] * dt
        
        # calcule de l'acceleration avec les conditions actuelles

        theta_dot_dot[i] = -g/l*math.sin(theta[i]) - a_c[i]/l*math.cos(theta[i]) - b/(m*l**2)*theta_dot[i]

        # calcul accélération, vitesse, position
        theta_dot[i+1] = theta_dot[i] + theta_dot_dot[i] * dt
        theta[i+1] = theta[i] + theta_dot[i] * dt



def graphiques():
    
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(t,theta, label="angular position")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t,theta_dot, label="angular speed")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t,theta_dot_dot, label="angular acceleration")
    plt.legend()
    plt.show()

def get_kinetic_energy(i):
    """ Calculates kinetic energy
    pre: i: int, current iteration
    post: float, kinetic energy
    """
    vel_c = np.array([v_c[i], 0])       # vitesse du chariot vecteur en 2 dimensions (x,y) -> (v_c, 0)
    speed = l * theta_dot[i]            # vitesse du pendule relative au chariot (scalaire)
    vel_rel_p = np.array([speed * math.sin(theta[i]), speed * math.cos(theta[i])])  # vitesse vectorielle
    vel_abs = vel_c + vel_rel_p         # vitesse absolue du pendule
    speed_abs = math.sqrt(vel_abs[0]**2 + vel_abs[1]**2)
    
    return 0.5 * m * speed_abs**2
    
def calculate_energy():
    for i in range(len(t)):
        e_pot[i] = m * g * l * (1-math.cos(theta[i]))   # mgh
        e_kin[i] = get_kinetic_energy(i) # speed of cart + speedc
        e_tot[i] = e_pot[i] + e_kin[i] 

def graph_energy():
    plt.figure(2)
    plt.plot(t, e_pot , label="potential")
    plt.plot(t, e_kin, label="kinetic")
    plt.plot(t, e_tot, label="total")
    plt.legend()
    plt.show()

def graph_phase():
    plt.figure(3)
    plt.plot(theta, theta_dot)
    plt.xlabel("angular position [rad]")
    plt.ylabel("angular velocity [rad/s]")
    plt.show()

### module
def simulate(motion):
    simulation(motion)
    calculate_energy()
    
### programme principal
if __name__ == '__main__':
    
    simulation("const")
    graphiques()
    calculate_energy()
    graph_energy()
    graph_phase()

