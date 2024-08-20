import numpy as np

class RPMD_integration:
    def __init__(self) -> None:
        pass

    @staticmethod
    def propagate_momenta_half(time_step, beads_momenta, beads_forces):
        ''' p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t) '''
        beads_momenta_half = beads_momenta + 0.5 * time_step * beads_forces
        return beads_momenta_half

    def transform_to_normal_mode(self, beads_momenta_half, beads_positions):
        ''' transform momenta and positions from coordinate representation to normal mode representation '''
        beads_momenta_half_k = self.coordinate_to_normal_mode_representation(beads_momenta_half)
        beads_positions_k = self.coordinate_to_normal_mode_representation(beads_positions)
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

    def transform_back_to_coordinates(self, beads_momenta_half_kp, beads_positions_kp):
        ''' back transform momenta and positions '''
        beads_momenta_half = self.normal_mode_to_coordinate_representation(beads_momenta_half_kp)
        beads_positions = self.normal_mode_to_coordinate_representation(beads_positions_kp)
        return beads_momenta_half, beads_positions

    @staticmethod
    def coordinate_to_normal_mode_representation(beads_vectors, C_jk, n_atom, n_beads):
        return np.array([[np.sum([beads_vectors[j,i,:]*C_jk[j,k] for j in range(n_beads)],axis=0) \
                          for i in range(n_atom)] for k in range(n_beads)])
    
    @staticmethod
    def normal_mode_to_coordinate_representation(beads_vectors, C_jk, n_atom, n_beads):
        return np.array([[np.sum([beads_vectors[j,i,:]*C_jk[k,j] for j in range(n_beads)],axis=0) \
                          for i in range(n_atom)] for k in range(n_beads)])
    