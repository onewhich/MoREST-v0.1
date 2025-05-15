import numpy as np

class MD_integration:
    def __init__(self, many_body_potential):
        self.many_body_potential = many_body_potential
    
    def velocity_Verlet(self, time_step, current_system, current_forces, masses):
        ### x(t), v(t) = p(t) / m
        current_positions = current_system.get_positions()
        #current_velocities = self.current_system.get_velocities()
        current_momenta = current_system.get_momenta()

        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * current_forces * time_step

        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        #next_positions = current_positions + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_positions = current_positions + (current_momenta * time_step + 0.5 * current_forces * time_step**2) / masses
        current_system.set_positions(next_positions)

        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(current_system)

        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * self.time_step
        #self.current_system.set_velocities(next_velocities)

        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        current_system.set_momenta(next_momenta)

        #next_velocities = next_system.get_velocities()

        return next_potential_energy, next_forces
    
    def Suzuki_Yoshida_4th(self, time_step, current_system, current_forces, masses):
        '''
        The 4th-order Suzuki–Yoshida symplectic integrator.
        Composes three 2nd-order velocity Verlet steps to achieve 4th-order accuracy.
        '''
        # Composition coefficient gamma = 1 / (2 - 2^(1/3))
        gamma = 1.3512071919596578
        # First Verlet step: step size gamma * time_step
        current_potential_energy, current_forces = self.velocity_Verlet(gamma * time_step, current_system, current_forces, masses)
        # Second Verlet step: step size (1 - 2gamma) * time_step
        current_potential_energy, current_forces = self.velocity_Verlet((1 - 2 * gamma) * time_step, current_system, current_forces, masses)
        # Third Verlet step: step size gamma * time_step
        next_potential_energy, next_forces = self.velocity_Verlet(gamma * time_step, current_system, current_forces, masses)
        
        return next_potential_energy, next_forces
    
    def Runge_Kutta_4th(self, time_step, current_system, current_forces, masses):
        '''
        This version comes from classic Runge-Kutta methods:
        Runge–Kutta methods. (2022, September 6). In Wikipedia. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
         https://www.haroldserrano.com/blog/visualizing-the-runge-kutta-method
        '''
        # x_1 = x_n, v_1 = v_n, a_1 = a_n
        x_1 = current_system.get_positions()
        v_1 = current_system.get_velocities()
        a_1 = current_forces/masses

        # x_2 = x_n + h/2 * v_1, v_2 = v_n + h/2 * a_1, a_2 = f(x_2)
        x_2 = x_1 + time_step/2 * v_1
        v_2 = v_1 + time_step/2 * a_1
        current_system.set_positions(x_2)
        Ep_2, F_2 = self.many_body_potential.get_potential_forces(current_system)
        a_2 = F_2/masses

        # x_3 = x_n + h/2 * v_2, v_3 = v_n + h/2 * a_2, a_3 = f(x_3)
        x_3 = x_1 + time_step/2 * v_2
        v_3 = v_1 + time_step/2 * a_2
        current_system.set_positions(x_3)
        Ep_3, F_3 = self.many_body_potential.get_potential_forces(current_system)
        a_3 = F_3/masses

        # x_4 = x_n + h * v_3, v_4 = v_n + h * a_3, a_4 = f(x_4)
        x_4 = x_1 + time_step * v_3
        v_4 = v_1 + time_step * a_3
        current_system.set_positions(x_4)
        Ep_4, F_4 = self.many_body_potential.get_potential_forces(current_system)
        a_4 = F_4/masses

        # x_n+1 = x_n + h/6 * (v_1 + 2*v_2 + 2*v_3 + v_4), v_n+1 = v_n + h/6 * (a_1 + 2*a_2 + 2*a_3 + a_4)
        next_coordinates = x_1 + time_step/6 * (v_1 + 2*v_2 + 2*v_3 + v_4)
        next_velocities = v_1 + time_step/6 * (a_1 + 2*a_2 + 2*a_3 + a_4)

        current_system.set_positions(next_coordinates)
        current_system.set_velocities(next_velocities)

        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(current_system)

        return next_potential_energy, next_forces

    @staticmethod
    def propagate_momenta_half(time_step, momenta, forces):
        ### p(t+0.5dt) = p(t) + 0.5 * F(t) * dt ; v(t+0.5dt) = p(t+0.5dt) / m
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        return momenta + 0.5 * forces * time_step

    @staticmethod
    def propagate_positions_half(time_step, positions, momenta, masses):
        ### x(t+0.5dt) = x(t) + 0.5*p(t)*dt / m
        return positions + 0.5 * momenta * time_step / masses
    
    @staticmethod
    def propagate_positions_p_half(time_step, positions, momenta_half, masses):
        ### x(t+dt) = x(t) + v(t+0.5dt)*dt
        return positions + momenta_half * time_step / masses

    @staticmethod
    def propagate_positions_p0(time_step, positions, momenta, forces, masses):
        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        return positions + (momenta * time_step + 0.5 * forces * time_step**2) / masses
    
    @staticmethod
    def propagate_random_velocities(time_step, velocities, gamma, kBT, noise=None):
        if noise == None:
            noise_vector = np.random.normal(size=np.shape(velocities))
            noise_vector = noise_vector / np.linalg.norm(noise_vector,axis=-1)[:,np.newaxis]
            noise = np.random.normal(size=(len(velocities),1)) * noise_vector
        c_1 = np.exp(-gamma*time_step)
        c_2 = np.sqrt(kBT*(1-c_1**2))
        next_velocities =  c_1*velocities+c_2*noise
        return next_velocities

class RPMD_integration(MD_integration):
    '''
    Implement RPMD in real space
    '''
    def __init__(self, many_body_potential, omega_n, n_beads):
        super().__init__(many_body_potential)
        self.omega_n = omega_n
        self.n_beads = n_beads
    
    def beads_harmonic_forces(self, beads_positions, masses):
        k = masses * self.omega_n**2

        # Use np.roll to get neighbor positions with periodic boundary conditions
        beads_positions_prev = np.roll(beads_positions, shift=1, axis=0)
        beads_positions_next = np.roll(beads_positions, shift=-1, axis=0)

        # Force: -k[(q - q_prev) + (q - q_next)]
        forces = -k * (2 * beads_positions - beads_positions_prev - beads_positions_next)

        return forces
    
    def RP_velocity_Verlet(self, time_step, current_beads, current_beads_forces, masses):
        beads_positions = np.array([i_bead.get_positions() for i_bead in current_beads])
        beads_momenta = np.array([i_bead.get_momenta() for i_bead in current_beads])
        
        RP_forces = current_beads_forces + self.beads_harmonic_forces(beads_positions, masses)

        beads_momenta_half = self.propagate_momenta_half(time_step, beads_momenta, RP_forces)

        next_beads_positions = self.propagate_positions_p0(time_step, beads_positions, beads_momenta, RP_forces, masses)

        for i in range(self.n_beads):
            current_beads[i].set_positions(next_beads_positions[i])
        beads_potential_energy = []
        next_beads_forces = []
        for i in range(self.n_beads):
            tmp_potential_energy, tmp_forces = self.many_body_potential.get_potential_forces(current_beads[i])
            beads_potential_energy.append(tmp_potential_energy)
            next_beads_forces.append(tmp_forces)
        beads_potential_energy = np.array(beads_potential_energy)
        next_beads_forces = np.array(next_beads_forces)

        next_RP_forces = next_beads_forces + self.beads_harmonic_forces(next_beads_positions, masses)
        next_beads_momenta = self.propagate_momenta_half(time_step, beads_momenta_half, next_RP_forces)
        for i in range(self.n_beads):
            current_beads[i].set_momenta(next_beads_momenta[i])

        return beads_potential_energy, next_beads_forces, next_beads_positions, next_beads_momenta

class RPMD_normal_mode_integration:
    '''
    Implement RPMD time evolution based on normal mode representation with modified velocity Verlet.
    @article{ceriotti2010efficient,
        title={Efficient stochastic thermostatting of path integral molecular dynamics},
        author={Ceriotti, Michele and Parrinello, Michele and Markland, Thomas E and Manolopoulos, David E},
        journal={The Journal of chemical physics},
        volume={133},
        number={12},
        year={2010},
        publisher={AIP Publishing}
    }
    '''
    def __init__(self, many_body_potential):
        self.many_body_potential = many_body_potential

    def RP_velocity_Verlet(self, time_step, current_beads, current_beads_forces, C_jk, n_atom, n_beads, omega_k, atom_masses):
        beads_positions = np.array([i_bead.get_positions() for i_bead in current_beads])
        beads_momenta = np.array([i_bead.get_momenta() for i_bead in current_beads])
        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = beads_momenta + 0.5 * time_step * current_beads_forces
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k = self.coordinate_to_normal_mode_representation(beads_momenta_half, C_jk, n_atom, n_beads)
        beads_positions_k = self.coordinate_to_normal_mode_representation(beads_positions, C_jk, n_atom, n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp = np.array([[np.cos(omega_k[k]*time_step)*beads_momenta_half_k[k,i,:] for i in range(n_atom)] \
                                    for k in range(n_beads)]) \
                                - np.array([[atom_masses[i]*omega_k[k]*np.sin(omega_k[k]*time_step)*beads_positions_k[k,i,:] \
                                    for i in range(n_atom)] for k in range(n_beads)])
        beads_positions_kp = np.array([[1/(atom_masses[i]*omega_k[k])*np.sin(omega_k[k]*time_step)*beads_momenta_half_k[k,i,:] \
                            for i in range(n_atom)] for k in range(n_beads)]) \
                            + np.array([[np.cos(omega_k[k]*time_step)*beads_positions_k[k,i,:] for i in range(n_atom)] \
                                for k in range(n_beads)])
        # back transform momenta and positions
        beads_momenta_half = self.normal_mode_to_coordinate_representation(beads_momenta_half_kp, C_jk, n_atom, n_beads)
        next_beads_positions = self.normal_mode_to_coordinate_representation(beads_positions_kp, C_jk, n_atom, n_beads)
        for i in range(n_beads):
            current_beads[i].set_positions(next_beads_positions[i])
        # calculate forces
        beads_potential_energy = []
        next_beads_forces = []
        for i in range(n_beads):
            tmp_potential_energy, tmp_forces = self.many_body_potential.get_potential_forces(current_beads[i])
            beads_potential_energy.append(tmp_potential_energy)
            next_beads_forces.append(tmp_forces)
        beads_potential_energy = np.array(beads_potential_energy)
        next_beads_forces = np.array(next_beads_forces)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = beads_momenta_half + 0.5 * time_step * next_beads_forces
        for i in range(n_beads):
            current_beads[i].set_momenta(next_beads_momenta[i])

        return beads_potential_energy, next_beads_forces, next_beads_positions, next_beads_momenta

    @staticmethod
    def propagate_momenta_half(time_step, beads_momenta, beads_forces):
        ''' p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t) '''
        beads_momenta_half = beads_momenta + 0.5 * time_step * beads_forces
        return beads_momenta_half

    def transform_to_normal_mode(self, beads_momenta_half, beads_positions, C_jk, n_atom, n_beads):
        ''' transform momenta and positions from coordinate representation to normal mode representation '''
        beads_momenta_half_k = self.coordinate_to_normal_mode_representation(beads_momenta_half, C_jk, n_atom, n_beads)
        beads_positions_k = self.coordinate_to_normal_mode_representation(beads_positions, C_jk, n_atom, n_beads)
        return beads_momenta_half_k, beads_positions_k

    @staticmethod
    def free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, omega_k, n_atom, n_beads, atom_masses):
        ''' exact evolution through dt under free ring polymer Hamiltonian '''
        beads_momenta_half_kp = np.array([[np.cos(omega_k[k]*time_step)*beads_momenta_half_k[k,i,:] for i in range(n_atom)] \
                                    for k in range(n_beads)]) \
                                - np.array([[atom_masses[i]*omega_k[k]*np.sin(omega_k[k]*time_step)*beads_positions_k[k,i,:] \
                                    for i in range(n_atom)] for k in range(n_beads)])
        beads_positions_kp = np.array([[1/(atom_masses[i]*omega_k[k])*np.sin(omega_k[k]*time_step)*beads_momenta_half_k[k,i,:] \
                            for i in range(n_atom)] for k in range(n_beads)]) \
                            + np.array([[np.cos(omega_k[k]*time_step)*beads_positions_k[k,i,:] for i in range(n_atom)] \
                                for k in range(n_beads)])
        return beads_momenta_half_kp, beads_positions_kp

    def transform_back_to_coordinates(self, beads_momenta_half_kp, beads_positions_kp, C_jk, n_atom, n_beads):
        ''' back transform momenta and positions '''
        beads_momenta_half = self.normal_mode_to_coordinate_representation(beads_momenta_half_kp, C_jk, n_atom, n_beads)
        beads_positions = self.normal_mode_to_coordinate_representation(beads_positions_kp, C_jk, n_atom, n_beads)
        return beads_momenta_half, beads_positions

    @staticmethod
    def coordinate_to_normal_mode_representation(beads_vectors, C_jk, n_atom, n_beads):
        return np.array([[np.sum([beads_vectors[j,i,:]*C_jk[j,k] for j in range(n_beads)],axis=0) \
                          for i in range(n_atom)] for k in range(n_beads)])
    
    @staticmethod
    def normal_mode_to_coordinate_representation(beads_vectors, C_jk, n_atom, n_beads):
        return np.array([[np.sum([beads_vectors[j,i,:]*C_jk[k,j] for j in range(n_beads)],axis=0) \
                          for i in range(n_atom)] for k in range(n_beads)])
    