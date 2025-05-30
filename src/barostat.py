import numpy as np
from ase import units
from wall_potential import repulsive_wall

class barostat_space:
    def __init__(self, barostat_parameters, current_system):
        self.barostat_parameters = barostat_parameters
        self.current_system = current_system
        self.n_atom = current_system.get_global_number_of_atoms()
        self.masses = current_system.self.get_masses()
        self.P_simulation = barostat_parameters['barostat_pressure']
        self.initialize_barostat_space_size()
        self.initialize_barostat_space_wall()

    def initialize_barostat_space_size(self):
        Eks = self.current_system.get_kinetic_energy()
        #Eks = self.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        forces_all = self.current_system.get_forces()
        coordinates_all = self.current_system.get_positions()
        for i in range(self.barostat_parameters['barostat_number']):
            index = self.barostat_parameters['barostat_action_atoms'][i]
            if index == 'all':
                index = np.arange(self.n_atom)
                self.barostat_parameters['barostat_action_atoms'][i] = index
            internal_virial = self.get_internal_virial(index, coordinates_all, forces_all)
            volume = 2*(Eks[i] - internal_virial)/(3*self.P_simulation[i])
            if self.barostat_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_parameters['barostat_space_size'].append(np.pow((3/4 * volume / np.pi), 1./3.))  # V = 4/3 * Pi * r^3; r = (3/4 * V/Pi)^(1/3)

    def initialize_barostat_space_wall(self):
        self.barostat_space_wall_parameters = {}
        self.barostat_space_wall_parameters['wall_number'] = 0
        self.barostat_space_wall_parameters['wall_collective_variable'] = []
        self.barostat_space_wall_parameters['wall_shape'] = []
        self.barostat_space_wall_parameters['wall_type'] = []
        self.barostat_space_wall_parameters['power_wall_direction'] = []
        self.barostat_space_wall_parameters['wall_scaling'] = []
        self.barostat_space_wall_parameters['wall_scope'] = []
        self.barostat_space_wall_parameters['wall_action_atoms'] = []
        self.barostat_space_wall_parameters['wall_shape_parameters'] = []
        for i, barostat_space in enumerate(self.barostat_parameters['barostat_space_parameters']):
            if self.barostat_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_space_wall_parameters['wall_number'] += 1
                self.barostat_space_wall_parameters['wall_collective_variable'].append(self.barostat_parameters['barostat_collective_variable'][i])
                self.barostat_space_wall_parameters['wall_shape'].append('spherical')
                self.barostat_space_wall_parameters['wall_type'].append('power_wall')
                self.barostat_space_wall_parameters['power_wall_direction'].append(-1)
                self.barostat_space_wall_parameters['wall_scaling'].append(1)
                self.barostat_space_wall_parameters['wall_scope'].append(2)
                self.barostat_space_wall_parameters['wall_action_atoms'].append(self.barostat_parameters['barostat_number'][i])
                tmp_parameters = {}
                tmp_parameters['spherical_wall_center'] = barostat_space['barostat_sphere_center']
                tmp_parameters['spherical_wall_radius'] = self.barostat_parameters['barostat_space_size'][i]
                self.barostat_space_wall_parameters['wall_shape_parameters'].append(tmp_parameters)
            elif self.barostat_parameters['barostat_space_shape'][i].upper() == 'cuboid'.upper():
                self.barostat_space_wall_parameters['wall_number'] += 6
                raise Exception('Cuboidal space has not been implemented yet.')
            elif self.barostat_parameters['barostat_space_shape'][i].upper() == 'plane'.upper():
                self.barostat_space_wall_parameters['wall_number'] += 1
                raise Exception('Planar space has not been implemented yet.')
            
        self.barostat_space_wall = repulsive_wall(self.barostat_space_wall_parameters)

    @staticmethod
    def get_atom_kinetic_energies(velocities, masses):
        """
        Calculate kinetic energy for each atom.

        Parameters:
            velocities (ndarray): shape (N, 3), velocities of N atoms
            masses (ndarray): shape (N, 1), masses of N atoms

        Returns:
            ndarray: shape (N, 1), kinetic energies of atoms
        """
        v_squared = np.sum(velocities ** 2, axis=1, keepdims=True)  # (N, 1)
        Eks = 0.5 * masses * v_squared  # (N, 1)
        return Eks

    @staticmethod
    def get_internal_virial(index_atom, coordinates_all, forces_all):
        """
        Approximate the virial contribution from a subset of atoms.

        Assumes total forces are available per atom (not per pairwise interaction).
        The virial is approximated as:
            W ≈ -sum_i (r_i · F_i), for i in index_atom

        Parameters:
            index_atom (array-like): indices of atoms in the group of interest
            coordinates_all (ndarray): shape (N, 3), positions of all atoms
            forces_all (ndarray): shape (N, 3), total forces on all atoms

        Returns:
            float: scalar virial of the specified group
        """
        index_atom = np.asarray(index_atom, dtype=int)
        if len(index_atom) == 0:
            return 0.0
        n_atom_all = coordinates_all.shape[0]
        if np.any(index_atom >= n_atom_all) or np.any(index_atom < 0):
            raise IndexError("index_atom contains out-of-bounds index")

        coords = coordinates_all[index_atom]   # shape (M, 3)
        forces = forces_all[index_atom]        # shape (M, 3)

        virial = -np.sum(np.einsum('ij,ij->i', coords, forces))
        return virial
    
    @staticmethod
    def get_internal_virial_tensor(index_atom, coordinates_all, forces_all):
        """
        Compute the virial tensor (3x3) for a group of atoms.

        Parameters:
            index_atom (array-like): indices of atoms in the group
            coordinates_all (ndarray): shape (N, 3), positions of all atoms
            forces_all (ndarray): shape (N, 3), forces on all atoms

        Returns:
            ndarray: (3, 3) virial tensor
        """
        index_atom = np.asarray(index_atom, dtype=int)
        if len(index_atom) == 0:
            return np.zeros((3, 3))
        
        if np.any(index_atom < 0) or np.any(index_atom >= coordinates_all.shape[0]):
            raise IndexError("index_atom contains out-of-bounds indices")
        
        coords = coordinates_all[index_atom]   # (M, 3)
        forces = forces_all[index_atom]        # (M, 3)

        # virial tensor: W_αβ = -Σ_i r_i^α F_i^β
        virial_tensor = -np.einsum('ia,ib->ab', coords, forces)  # shape (3, 3)

        return virial_tensor


    @staticmethod    
    def get_volume(space_shape: str, space_size) -> float:
        """
        Compute volume of a defined simulation region.

        Parameters:
            space_shape: str, shape of the region ('sphere', 'cuboid', 'plane')
            space_size: 
                - float (for 'sphere'): radius
                - tuple of 3 floats (for 'cuboid'): (length, width, height)

        Returns:
            float: Volume of the space

        Raises:
            ValueError: if the shape or parameters are invalid
            NotImplementedError: if the shape is recognized but not yet implemented
        """
        shape = space_shape.lower()
        if shape == 'sphere':
            if not isinstance(space_size, (int, float)):
                raise ValueError("For 'sphere', space_size should be a number (radius).")
            radius = space_size
            volume = (4.0 / 3.0) * np.pi * radius ** 3 # V = 4/3 * Pi * r^3
        elif shape == 'cuboid':
            if not (isinstance(space_size, (tuple, list)) and len(space_size) == 3):
                raise ValueError("For 'cuboid', space_size must be a tuple/list of 3 values (length, width, height).")
            length, width, height = space_size
            volume = length * width * height
        elif shape == 'plane':
            raise NotImplementedError("Volume calculation for 'plane' shape requires cell-based context and is not implemented yet.")
        else:
            raise ValueError(f"Unsupported space shape: '{space_shape}'")
        return volume
    
    @staticmethod
    def get_pressure(Ek_atoms, internal_virial, volume):
        """
        Compute the instantaneous pressure of the system.

        Parameters:
            Ek_atoms: ndarray of per-atom kinetic energies
            internal_virial: scalar internal virial term (∑ r·F)
            volume: system volume

        Returns:
            Instantaneous pressure
        """
        return (np.sum(Ek_atoms) - internal_virial)*2/(3 * volume)

    def update_barostat_space_wall(self):
        for i in range(self.barostat_parameters['barostat_number']):
            if self.barostat_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_space_wall_parameters['wall_shape_parameters'][i]['spherical_wall_radius'] = \
                                                                                self.barostat_parameters['barostat_space_size'][i]
        self.barostat_space_wall.update_wall_parameters(self.barostat_space_wall_parameters)

    def get_barostat_space_bias_forces(self):
        coordinates_all = self.current_system.get_positions()
        barostat_bias_forces = np.zeros((self.n_atom,3))
        for i in range(self.barostat_parameters['barostat_number']):
            index = self.barostat_parameters['barostat_action_atoms'][i]
            coordinates = coordinates_all[index]
            tmp_bias = np.array([self.barostat_space_wall.get_repulsive_wall_force(i_coordinate) for i_coordinate in coordinates])
            for j, j_bias in enumerate(tmp_bias):
                barostat_bias_forces[index[j]] += j_bias
        return barostat_bias_forces

def Berendsen_volume_rescaling(barostat_parameters, time_step, coordinates_all, forces_all, velocities, masses, P_simulation, tau_P, factor_Z):
    Eks = barostat_space.get_atom_kinetic_energies(velocities, masses)
    center_of_mass = (coordinates_all*masses).sum(axis=0) / masses.sum()
    P_current = []
    for i in range(barostat_parameters['barostat_number']):
        index_atom = barostat_parameters['barostat_action_atoms'][i]
        internal_virial = barostat_space.get_internal_virial(index_atom, coordinates_all, forces_all)
        volume = barostat_space.get_volume(barostat_parameters['barostat_space_shape'][i], barostat_parameters['barostat_space_size'][i])
        P_current.append(barostat_space.get_pressure(Eks[index_atom], internal_virial, volume))
        factor_mu = 1-time_step*factor_Z/tau_P/3.*(P_simulation[i]-P_current[i])
        # It can become unstable if:
        # The time step (time_step) is too large,
        # The pressure relaxation time (tau_P) is too short,
        # The pressure difference is too big.
        # This might result in tmp_miu becoming negative or unreasonably small/large, causing unphysical behavior or numerical divergence.
        # To ensure numerical stability, add clamping:
        factor_mu = max(0.9, min(1.1, factor_mu))

        if barostat_parameters['barostat_space_type'][i].upper() == 'equilibrium'.upper():
            barostat_parameters['barostat_space_size'][i] *= factor_mu
            new_coordinates = (coordinates_all[index_atom] - center_of_mass) * factor_mu + center_of_mass
            coordinates_all[index_atom] = new_coordinates
        elif barostat_parameters['barostat_space_type'][i].upper() == 'ultrafast'.upper():
            barostat_parameters['barostat_space_size'][i] *= factor_mu

    return coordinates_all, np.array(P_current)
    
def stochastic_velocity_volume_rescaling(barostat_parameters, time_step, half_time_step, coordinates_all, forces_all, velocities, eta, momenta, \
                                         masses, W_barostat, T_current, P_simulation):
    Eks = barostat_space.get_atom_kinetic_energies(velocities, masses)
    P_current = []
    for i in range(barostat_parameters['barostat_number']):
        index_atom = barostat_parameters['barostat_action_atoms'][i]

        # stage 2
        internal_virial = barostat_space.get_internal_virial(index_atom, coordinates_all, forces_all)
        volume = barostat_space.get_volume(barostat_parameters['barostat_space_shape'][i], barostat_parameters['barostat_space_size'][i])
        P_current.append(barostat_space.get_pressure(Eks[index_atom], internal_virial, volume))
        eta_half, momenta_half = SVR_stage_2_4(eta[i], volume, P_current[i], P_simulation[i], T_current, half_time_step, W_barostat, \
                                               forces_all[index_atom], momenta[index_atom], masses[index_atom], index_atom)
        # stage 3
        eta_half_time_step = eta_half*time_step
        volume *= np.exp(3*eta_half_time_step)
        momenta_half *= np.exp(-1*eta_half_time_step)
        if barostat_parameters['barostat_space_type'][i].upper() == 'gas'.upper():
            barostat_parameters['barostat_space_size'][i] *= np.exp(eta_half_time_step)
        elif barostat_parameters['barostat_space_type'][i].upper() == 'condensed'.upper():
            barostat_parameters['barostat_space_size'][i] *= np.exp(eta_half_time_step)
            coordinates_all[index_atom] = np.exp(eta_half_time_step)*coordinates_all[index_atom] + \
                                     np.sinh(eta_half_time_step)/eta_half * \
                                     (momenta_half/masses[index_atom])
        # stage 4
        internal_virial = barostat_space.get_internal_virial(index_atom, coordinates_all, forces_all)
        P_current[i] = barostat_space.get_pressure(0.5*momenta_half**2/masses[index_atom], internal_virial, volume)
        new_eta, new_momenta = SVR_stage_2_4(eta_half, volume, P_current[i], P_simulation[i], T_current, half_time_step, W_barostat, \
                                               forces_all[index_atom], momenta_half, masses[index_atom], index_atom)

        eta[i] = new_eta
        momenta[index_atom] = new_momenta

    return coordinates_all, momenta, eta, P_current

def SVR_stage_2_4(eta, volume, P_current, P_simulation, T_current, half_time_step, W_barostat, forces, momenta, masses, index):
    eta_half = eta + 3*(volume*(P_current-P_simulation)+2*units.kB*T_current)*half_time_step/W_barostat + \
                        np.sum([np.dot(forces[i],momenta[i])/masses[i] for i in range(len(index))])/W_barostat*half_time_step**2 + \
                        np.sum([np.dot(forces[i],forces[i])/masses[i] for i in range(len(index))])/W_barostat/3*half_time_step**3
    momenta_half = momenta + forces * half_time_step
    return eta_half, momenta_half
