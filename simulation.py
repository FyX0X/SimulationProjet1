import math
import numpy as np


class Pendulum:
    """
    Representation of a pendulum on a cart's state with respect to time.

    Can be used to store a simulated pendulum or simply experimental data.

    Class Attributes:
        g (float): Gravitational acceleration of the pendulum's environment [m/s^2]

    Instance Attributes:
        name (str): A name to identify the pendulum (for graphing purpose).
        time_shift (float): How much should the graph be horizontally shifted (for graphing purpose).

        m (float): Mass of the pendulum [kg].
        l (float): Length of the pendulum [m].

        time (np.ndarray): A 1D array of time points [s] where the pendulum's state is evaluated.

        theta (np.ndarray): Angular position of the pendulum at each time step [rad].
        omega (np.ndarray): Angular velocity of the pendulum at each time step [rad/s].
        alpha (np.ndarray): Angular acceleration of the pendulum at each time step [rad/s^2].

        x_c (np.ndarray): Position of the cart of the pendulum at each time step [rad].
        v_c (np.ndarray): Velocity of the cart of the pendulum at each time step [rad/s].
        a_c (np.ndarray): Acceleration of the cart of the pendulum at each time step [rad/s^2].

    """

    g = 9.81

    def __init__(self, name: str, mass: float, length: float, time: np.ndarray, theta: np.ndarray = None):
        """
        Initialize the Pendulum object with a given time array.

        Args:
            name (str): A name to indentify the pendulum
            mass (float): Mass of the pendulum [kg]
            length (float): Length of the pendulum [m]
            time (np.ndarray): A 1D array of time points [s] where the pendulum's state is evaluated.
            theta (np.ndarray, optional): A 1D array of theta points [rad] of the pendulum at each time step.
                If provided, angular velocity (`omega`) and angular acceleration (`alpha`) will be calculated.


        Initializes the `theta`, `omega`, `alpha`, `x_c`, `v_c`, and `a_c` arrays based on the provided
        `time` array. These arrays represent the angular position (`theta`), angular velocity (`omega`),
        and angular acceleration (`alpha`) of the pendulum, as well as the position (`x_c`), velocity (`v_c`),
        and acceleration (`a_c`) of the cart at each time step, respectively.
        """

        self.name = name
        self.time_shift = 0

        self.m = mass
        self.l = length

        self.time = time

        self.theta = np.zeros_like(time)
        self.omega = np.zeros_like(time)
        self.alpha = np.zeros_like(time)

        self.x_c = np.zeros_like(time)
        self.v_c = np.zeros_like(time)
        self.a_c = np.zeros_like(time)

        # calculates angular speed and acceleration if from data
        if theta is not None:
            self.theta = theta
            self.derive_angular_state()

    def derive_angular_state(self) -> None:
        """
        Derive the cart's angular velocity and acceleration from the angular position.

        The angular velocity is calculated as the time derivative of the angular position, and the angular acceleration
        is calculated as the time derivative of the angular velocity.

        This function assumes that `theta` contains the position of the cart at each time point.
        The resulting angular velocity (`omega`) and acceleration (`alpha`) are stored in the respective arrays.

        Boundary conditions:
            - The angular velocity at the first time step is set equal to the second time step.
            - The angular acceleration at the first time step is set to zero.
        """

        dt = self.time[-1]/(len(self.time) - 1)
        # Calculate angular velocity
        for i in range(len(self.time)-1):
            self.omega[i+1] = (self.theta[i+1] - self.theta[i]) / dt
        # Set boundary condition
        self.omega[0] = self.omega[1]

        # calculate acceleration
        for i in range(len(self.time)-1):
            self.alpha[i+1] = (self.omega[i+1] - self.omega[i]) / dt
        # Set boundary condition
        self.alpha[0] = 0

    def derive_cart_state_from_position(self) -> None:
        """
        Derive the cart's velocity and acceleration from the given position.

        The velocity is calculated as the time derivative of the position, and the acceleration
        is calculated as the time derivative of the velocity.

        This function assumes that `x_c` contains the position of the cart at each time point.
        The resulting velocity (`v_c`) and acceleration (`a_c`) are stored in the respective arrays.

        Boundary conditions:
            - The velocity at the first time step is set equal to the second time step.
            - The acceleration at the first time step is set to zero.
        """

        dt = self.time[-1]/(len(self.time) - 1)

        # Calculate velocity
        for i in range(len(self.time)-1):
            self.v_c[i+1] = (self.x_c[i+1] - self.x_c[i]) / dt
        # Set boundary condition
        self.v_c[0] = self.v_c[1]

        # calculate acceleration
        for i in range(len(self.time)-1):
            self.a_c[i+1] = (self.v_c[i+1] - self.v_c[i]) / dt
        # Set boundary condition
        self.a_c[0] = 0

    def integrate_cart_state_from_acceleration(self) -> None:
        """
        Integrate the cart's acceleration to obtain its position and velocity.

        Given the cart's acceleration (`a_c`), this function integrates the acceleration to
        obtain the velocity and then integrates the velocity to obtain the position over time.

        Boundary conditions:
            - The initial position (`x_c[0]`) and velocity (`v_c[0]`) are set to zero.
        """

        dt = self.time[-1]/(len(self.time)-1)

        # Set initial conditions for position and velocity
        self.x_c[0] = 0
        self.v_c[0] = 0

        # Integrate acceleration to get velocity and position
        for i in range(len(self.time)-1):
            self.v_c[i+1] = self.v_c[i] + self.a_c[i] * dt
            self.x_c[i+1] = self.x_c[i] + self.v_c[i] * dt

    def get_kinetic_energy(self) -> np.ndarray:
        """
        Calculate the kinetic energy of the pendulum and cart system.

        Steps:
            1. The cart's velocity is represented as a 2D vector (v_x, 0), where `v_x` is the cart's
               velocity in the x-direction, and there's no velocity in the y-direction.
            2. The pendulum's relative velocity is computed from its angular velocity (`omega`) and
               position (`theta`), converting it to Cartesian coordinates.
            3. The total velocity of the pendulum is the sum of the cart's velocity and the pendulum's
               relative velocity.
            4. The kinetic energy is calculated using the formula:
               KE = 0.5 * m * v^2, where v^2 is the squared magnitude of the velocity.

        Returns:
            np.ndarray: An array containing the kinetic energy at each time step.
        """

        # Create the 2D velocity for the cart (cart is only moving in x-direction)
        cart_velocity = np.column_stack((self.v_c, np.zeros_like(self.v_c)))  # (v_x, 0) for cart velocity

        # Relative speed of pendulum at the end (angular velocity * length)
        pendulum_relative_speed = self.omega * self.l

        # Calculate the pendulum's velocity components in x and y (relative to the cart)
        speed_x = pendulum_relative_speed * np.sin(self.theta)
        speed_y = pendulum_relative_speed * np.cos(self.theta)
        pendulum_relative_velocity = np.column_stack((speed_x, speed_y))

        # Total velocity (cart + pendulum) in 2D
        pendulum_absolute_velocity = cart_velocity + pendulum_relative_velocity

        # Calculate the speed squared for each time step (magnitude squared of velocity vector)
        speed_squared = np.sum(np.square(pendulum_absolute_velocity), axis=1)

        # Kinetic energy: KE = 1/2 * m * v^2 (where v^2 is the squared speed)
        return 0.5 * self.m * speed_squared

    def get_potential_energy(self) -> np.ndarray:
        """Calculate the gravitational potential energy of the pendulum at each time step."""

        return self.m * self.g * self.l * (1 - np.cos(self.theta))

    def get_total_energy(self) -> np.ndarray:
        """Calculate the total mechanical energy of the pendulum."""

        return self.get_potential_energy() + self.get_kinetic_energy()


class SimulatedPendulum(Pendulum):
    """
    A simulated pendulum attached to a cart.

    This class extends the basic Pendulum model by adding the simulation methods to calculate the state of the pendulum.
    The simulation includes the friction as an additional physical parameter.

    Class Attributes:
        dt (float): Time step of the simulation [s].

    Instance Attributes:
        b (float): Friction coefficient of the pendulum [kg*m^2/s].
    """

    dt = 0.0005

    def __init__(self, name: str, mass: float, length: float, theta_0: float, omega_0: float, friction_coefficient: float, end: float = 30):
        """
        Initialize and simulate the pendulum.

        Args:
            name (str): A name to indentify the pendulum.
            mass (float): Mass of the pendulum [kg].
            length (float): Length of the pendulum [m].
            theta_0 (float): Initial angular position of the pendulum [rad].
            omega_0 (float): Initial angular velocity of the pendulum [rad/s].
            friction_coefficient (float): Friction coefficient of the pendulum [kg*m^2/s].
            end (float, optional): Length of the simulation [s]. Defaults to 30[s]
        """

        super().__init__(name, mass, length, np.arange(0, end, SimulatedPendulum.dt))
        self.b = friction_coefficient

        self.theta[0] = theta_0
        self.omega[0] = omega_0

        # simulates the pendulum evolution
        self.simulate()

    def simulate(self) -> None:
        """
        Runs the simulation of the pendulum using Euler's Integration method.

        This method calculates and stores the angular position, velocity, and acceleration
        of the pendulum with respect to time. The results are stored in the `theta`,
        `omega`, and `alpha` arrays, respectively.

        Updates:
            alpha: Angular acceleration of the pendulum at each time step.
            omega: Angular velocity of the pendulum at each time step.
            theta: Angular position of the pendulum at each time step.
        """

        for i in range(len(self.time)-1):
            self.alpha[i] = - self.g / self.l * math.sin(self.theta[i]) \
                            - self.a_c[i] / self.l * math.cos(self.theta[i]) \
                            - self.b / (self.m * self.l ** 2) * (self.omega[i])

            self.omega[i + 1] = self.omega[i] + self.alpha[i] * self.dt
            self.theta[i + 1] = self.theta[i] + self.omega[i] * self.dt

    def set_cart_profile_from_position(self, movement_profile: str, pulsation: float = 1, amplitude: float = 1) -> None:
        """
        Set the cart's position profile based on the specified movement type.

        Args:
            movement_profile (str): Type of movement profile for the cart. Options are "const", "sinus", "triangle".
            pulsation (float): The pulsation (frequency) of the movement profile in radians per second.
            amplitude (float): The amplitude of the movement profile.

        Sets the cart's position (`x_c`), velocity (`v_c`), and acceleration (`a_c`) based on the chosen profile.
        """

        # Set position profile based on the movement type
        match movement_profile:
            case "const":
                self.x_c = np.zeros_like(self.time)
            case "sinus":
                self.x_c = amplitude * np.sin(self.time * pulsation)
            case default:
                raise ValueError(f"Invalid movement profile: {movement_profile}")

        self.derive_cart_state_from_position()

    def set_cart_profile_from_acceleration(self, movement_profile: str, pulsation: float = 1, amplitude: float = 1) -> None:
        """
        Set the cart's acceleration profile based on the specified movement type.

        Args:
            movement_profile (str): Type of acceleration profile for the cart. Options are "const", "sinus", "triangle", "square".
            pulsation (float): The pulsation (frequency) of the movement profile in radians per second.
            amplitude (float): The amplitude of the movement profile.

        Sets the cart's position (`x_c`), velocity (`v_c`), and acceleration (`a_c`) based on the chosen profile.
        """

        # Set acceleration profile based on the movement type
        match movement_profile:
            case "const":
                self.a_c = np.zeros_like(self.time)
            case "sinus":
                self.a_c = amplitude * np.sin(self.time * pulsation)
            case "triangle":
                self.a_c = amplitude * (2 * np.abs(np.mod(self.time * pulsation * np.pi, 1) - 0.5) - 1)
            case "square":
                self.a_c = amplitude * np.sign(np.sin(self.time * pulsation))
            case default:
                raise ValueError(f"Invalid movement profile: {movement_profile}")

        self.integrate_cart_state_from_acceleration()
