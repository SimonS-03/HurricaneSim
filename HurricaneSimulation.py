import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

Temp = 273.15 + 10 # Kelvin
class Fluid:
  def __init__(self, T = Temp, visk = 2.791*10**(-7)*Temp**(0.7335), dens = 1.2):
    self.T = T
    self.viskositet = visk
    self.densitet = dens
    k_viskositet = self.viskositet/self.densitet #J/kg*s alt: m^2/s

class Orkan():

	def __init__(self, U:float, a:float, phi_n:int, r_max:int, r_step:int, t_max:int, t_step:int):
		self.r0 = a
		self.U = U
		self.phi_n = phi_n
		self.r_max = r_max
		self.t_max = t_max
		self.r_step = r_step
		self.t_step = t_step
		self.t_list = [0]
		self.r_list = []
		#Create initial r_list
		for r in range(self.r0, self.r_max, self.r_step):
			self.r_list.append(Velocity(r, 0, dt=self.t_step, dr=self.r_step))
		self.rv_t_matrix = []

	def time_step(self):
		for i in range(len(self.r_list)):
			if i < len(self.r_list)-2:
				self.r_list[i].integrate_t(self.r_list[i+1].vector, self.r_list[i+2].vector)
			elif i < len(self.r_list)-1:
				self.r_list[i].integrate_t(self.r_list[i+1].vector, [0, 0])
			else:
				self.r_list[i].integrate_t([0, 0], [0, 0])
		self.rv_t_matrix.append([objekt.vector.copy() for objekt in self.r_list])


	def create_v_t_matrix(self):
		self.rv_t_matrix.append([objekt.vector.copy() for objekt in self.r_list])
		for i in range(1, int(self.t_max/self.t_step), 1):
			t = self.t_step * i
			self.time_step()
			self.t_list.append(t)
			
class Velocity:
    def __init__(self, r, phi, dt, dr, dp_const=128, rotation=0.0002, latitude=np.pi/4):
        self.vector = [0, 0]
        self.r = r
        self.phi = phi
        T = 273.15 + 20 # Kelvin
        self.fluid = Fluid()
        self.dp_const = dp_const # Pressure gradiant
        self.earth_rotation = rotation
        self.latitude = latitude
        self.dt = dt
        self.dr = dr

    def integrate_t(self, next_vel, next2_vel):
        dr = self._durdt(next_vel[0], next2_vel[0])*self.dt
        self.vector[0] = self.vector[0] + dr
        self.vector[1] = self.vector[1] + self._dupdt(next_vel[1], next2_vel[1])*self.dt

    def _durdt(self, next_ur, next_ur2) -> float:
        cur_ur = self.vector[0]
        A = 2
        B = 0.5

        # Navier-stokes med coreolis-acceleration
        grad_vel_term = -cur_ur*((next_ur-cur_ur)/self.dr)
        grad_p_term =  - (1/self.fluid.densitet) * A * self.dp_const * math.exp(-A/self.r**B) / (self.r**(B+1)) 
        coreolis_term = self._coreolis_force()[0]
        laplace_term = self.fluid.viskositet * (next_ur2-2*next_ur+cur_ur)/(2*self.dr)

        return grad_vel_term + grad_p_term + coreolis_term + laplace_term


    def _dupdt(self, next_up, next_up2):
        cur_up = self.vector[1]

        # Navier-stokes med coreolis-acceleration
        grad_vel_term = 0
        grad_p_term = 0
        coreolis_term = self._coreolis_force()[1]
        laplace_term = self.fluid.viskositet * (next_up2-2*next_up+cur_up)/(2*self.dr)

        return grad_vel_term + grad_p_term + coreolis_term + laplace_term

    def _coreolis_force(self) -> list[float, float]:
       scaling = 2*self.earth_rotation*math.sin(self.latitude)
       return [scaling*(-self.vector[1]), scaling*self.vector[0]]


class SphericalVectorFieldSimulation:
    def __init__(self, velocity_matrix, time_list, r_values, phi_n=36, orkan: Orkan = None):
        """
        Initialize the simulation.

        Parameters:
        - velocity_matrix: A 2D list or numpy array of shape (len(time_list), len(r_values), 2)
          containing velocities [vr, vphi] at each time step for each radial value.
        - time_list: A list of time values.
        - r_values: A list of radial distances.
        - phi_n: Number of angular divisions for phi (default is 36).
        """
        self.orkan: Orkan = orkan
        self.velocity_matrix = np.array(velocity_matrix)
        self.time_list = time_list
        self.r_values = r_values
        self.phi_n = phi_n
        self.phi_values = np.linspace(0, 2 * np.pi, phi_n, endpoint=False)

    def generate_positions_and_velocities(self, t_index):
        """
        Generate Cartesian positions and velocity vectors for the given time index.

        Parameters:
        - t_index: Index of the current time step.

        Returns:
        - positions: List of (x, y) positions.
        - velocities: List of (vx, vy) velocities.
        """
        positions = []
        velocities = []

        for r_idx, r in enumerate(self.r_values):
            vr, vphi = self.velocity_matrix[t_index, r_idx]  # Spherical velocities
            for phi in self.phi_values:
                scale = 1
				# Convert spherical to Cartesian
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                vx = (vr * np.cos(phi) - vphi * np.sin(phi)) * scale
                vy = (vr * np.sin(phi) + vphi * np.cos(phi)) * scale

                positions.append((x, y))
                velocities.append((vx, vy))

        return positions, velocities


    def animate_vector_field(self):
        """
        Create and display an animation of the vector field with constant-sized arrows
        and a fixed colormap range from 0 to 0.2.
        """
        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.set_xlim(-max(self.r_values), max(self.r_values))
        ax.set_ylim(-max(self.r_values), max(self.r_values))

        # Initialize quiver plot
        positions, velocities = self.generate_positions_and_velocities(0)
        X, Y = zip(*positions)
        U, V = zip(*velocities)

        # Calculate magnitudes
        magnitudes = np.sqrt(np.array(U)**2 + np.array(V)**2)

        # Fix the colormap range
        vmin = 0
        vmax = 15 # Fixed maximum value
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = plt.cm.plasma  # Choose a colormap

        # Normalize velocities for constant arrow size
        U_normalized = np.array(U) / (magnitudes + 1e-6)
        V_normalized = np.array(V) / (magnitudes + 1e-6)

        quiver = ax.quiver(
            X, Y, U_normalized, V_normalized,
            magnitudes,  # Color based on magnitude
            cmap=cmap,
            norm=norm,
            scale=20,  # Adjust scale for arrow size
        )

        # Add color bar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Velocity Magnitude')

        def update(frame):
            t_index = frame % len(self.time_list)  # Wrap around if frame exceeds the time list
            positions, velocities = self.generate_positions_and_velocities(t_index)
            X, Y = zip(*positions)
            U, V = zip(*velocities)

            # Calculate magnitudes and normalize velocities
            magnitudes = np.sqrt(np.array(U)**2 + np.array(V)**2)
            U_normalized = np.array(U) / (magnitudes + 1e-6)
            V_normalized = np.array(V) / (magnitudes + 1e-6)

            quiver.set_offsets(np.c_[X, Y])
            quiver.set_UVC(U_normalized, V_normalized, magnitudes)  # Update colors with magnitudes
            quiver.set_array(magnitudes)  # Explicitly update colors
            return quiver,

        particle_n = 100
        particle_list = []
        for k in range(5):
            append_list = [Particle(orkan = self.orkan, r0 = self.orkan.r0 + i*(self.orkan.r_max-self.orkan.r0)/particle_n, phi0 = k*2*np.pi/5, dt = self.orkan.t_step) for i in range(particle_n)]
            particle_list.extend(append_list)
        
        def update_particle_positions():
            positions = []
            for particle in particle_list:
                if particle.dt != 0:
                    particle.integrate()
                particle.timestep += 1
                positions.append((particle.r*np.cos(particle.phi), particle.r*np.sin(particle.phi)))
            #print(positions)
            return positions
        
        def update_particles(frame):
             A, B = zip(*update_particle_positions())
             scatter.set_offsets(np.c_[A, B])
             return scatter

        A, B = zip(*update_particle_positions())
        
        scatter = ax.scatter(
             A, B
        )

        # Animation
        ani = FuncAnimation(fig, update, frames=len(self.time_list), interval=1, blit=False)
        ani2 = FuncAnimation(fig, update_particles, frames=len(self.time_list), interval = 1, blit = False)
        plt.show()

class Particle:
    def __init__(self, orkan: Orkan, size = 1, r0 = 0, phi0 = 0, dt = 1):
        self.size = size
        self.r = r0
        self.phi = phi0
        self.orkan = orkan
        self.dt = dt
        self.timestep = 0
        self.rt_array = []#np.array(int(orkan.t_max/orkan.t_step))
        
        """
        for step in range(int(orkan.t_max/orkan.t_step)):
            self.integrate()
        """
 
    def integrate(self):
        r_index = int((self.r-self.orkan.r0) % self.orkan.r_step)
        scaling = 1
        self.r = self.r + self.orkan.rv_t_matrix[self.timestep][r_index][0]*self.dt*scaling
        if self.r < self.orkan.r0: self.r = self.orkan.r_max
        
        self.phi = self.phi + self.orkan.rv_t_matrix[self.timestep][r_index][1]*self.dt*scaling/self.r
        self.phi = self.phi % (2*np.pi)
        self.rt_array.append([self.r, self.phi])
         
         
    
    

if __name__ == "__main__":
    #example()
    
    phi_n = 100
    #o = Orkan(U=1, a=30*10**3, phi_n=phi_n, r_max=200*10**3, t_max=1000, t_step=1, r_step=10*10**3)
    o = Orkan(U=1, a=100, phi_n=phi_n, r_max=1000, t_max=20000, t_step=5, r_step=20)
    o.create_v_t_matrix()
    r_values = [objekt.r for objekt in o.r_list]
    sim = SphericalVectorFieldSimulation(o.rv_t_matrix, o.t_list, r_values, phi_n, orkan = o)
    sim.animate_vector_field()
    
