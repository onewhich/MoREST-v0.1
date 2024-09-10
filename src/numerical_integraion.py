import numpy as np

class MD_integration:
    def __init__(self, many_body_potential):
        self.many_body_potential = many_body_potential
    
    def velocity_Verlet(self, time_step, current_system, current_forces, masses):
        ### x(t), v(t) = p(t) / m
        current_positions = current_system.get_positions()
        #current_velocities = self.current_system.get_velocities()
        current_momenta = current_system.get_momenta()

        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        #next_positions = current_positions + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_positions = current_positions + (current_momenta * time_step + 0.5 * current_forces * time_step**2) / masses
        current_system.set_positions(next_positions)

        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * current_forces * time_step

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
    
    @staticmethod
    def propagate_momenta_half(time_step, momenta, forces):
        ### p(t+0.5dt) = p(t) + 0.5 * F(t) * dt ; v(t+0.5dt) = p(t+0.5dt) / m
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        return momenta + 0.5 * forces * time_step

    @staticmethod
    def propagate_positions(time_step, positions, momenta, forces, masses):
        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        return positions + (momenta * time_step + 0.5 * forces * time_step**2) / masses

    @staticmethod
    def propagate_positions_half(time_step, positions, momenta, masses):
        ### x(t+0.5dt) = x(t) + 0.5*p(t)*dt / m
        return positions + 0.5 * momenta * time_step / masses
    
    @staticmethod
    def propagate_random_velocities(time_step, velocities, gamma, kBT, noise=None):
        if noise == None:
            noise = np.random.normal(size=np.shape(velocities))
        c_1 = np.exp(-gamma*time_step)
        c_2 = np.sqrt(kBT*(1-c_1*c_1))
        next_velocities =  c_1*velocities+c_2*noise
        return next_velocities

class RPMD_integration:
    '''
    Implement RPMD in real space
    '''
    def __init__(self) -> None:
        pass

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
    def __init__(self) -> None:
        pass

    def RP_velocity_Verlet(self, time_step, beads_positions, beads_momenta, beads_forces, C_jk, n_atom, n_beads, omega_k, atom_masses):
        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = beads_momenta + 0.5 * time_step * beads_forces
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
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = beads_momenta_half + 0.5 * time_step * beads_forces

        return next_beads_positions, next_beads_momenta

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
    