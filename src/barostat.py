import numpy as np
from ase import units
from wall_potential import repulsive_wall

class barostat_space:
    def __init__(self, barostat_parameters, current_system):
        self.barostat_parameters = barostat_parameters
        self.current_system = current_system
        self.n_atom = current_system.get_global_number_of_atoms()
        self.masses = current_system.get_masses()[:,np.newaxis]
        self.P_simulation = np.zeros(self.barostat_parameters['barostat_number'])
        for i in range(self.barostat_parameters['barostat_number']):
            self.P_simulation[i] = self.barostat_parameters['barostat_pressure'][i]
            index = self.barostat_parameters['barostat_action_atoms'][i]
            if index == 'all':
                index = np.arange(self.n_atom)
                self.barostat_parameters['barostat_action_atoms'][i] = index
        self.P_simulation = np.array(self.P_simulation)
        #self.initialize_barostat_space_size()
        self.initialize_barostat_space_wall()

    def initialize_barostat_space_size(self):
        Eks = self.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        forces_all = self.current_system.get_forces()
        coordinates_all = self.current_system.get_positions()
        for i in range(self.barostat_parameters['barostat_number']):
            index = self.barostat_parameters['barostat_action_atoms'][i]
            internal_virial = self.get_internal_virial(index, coordinates_all, forces_all)
            volume = 2*(np.sum(Eks[index]) - internal_virial)/(3*self.P_simulation[i])
            if self.barostat_parameters['barostat_space_shape'][i].lower() == 'sphere':
                self.barostat_parameters['barostat_space_size'].append(np.power((3*volume)/(4*np.pi), 1./3.))  # V = 4/3 * Pi * r^3; r = (3V/(4Pi))^(1/3)
            else:
                raise ValueError(f"Unsupported space shape: '{self.barostat_parameters['barostat_space_shape'][i]}'")

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
            self.barostat_space_wall_parameters['wall_number'] += 1
            self.barostat_space_wall_parameters['wall_collective_variable'].append(self.barostat_parameters['barostat_collective_variable'][i])
            self.barostat_space_wall_parameters['wall_type'].append('power_wall')
            self.barostat_space_wall_parameters['power_wall_direction'].append(1)
            self.barostat_space_wall_parameters['wall_scaling'].append(1)
            self.barostat_space_wall_parameters['wall_scope'].append(4)
            self.barostat_space_wall_parameters['wall_action_atoms'].append(self.barostat_parameters['barostat_action_atoms'][i])
            if self.barostat_parameters['barostat_space_shape'][i].lower() == 'sphere':
                self.barostat_space_wall_parameters['wall_shape'].append('spherical')
                tmp_parameters = {}
                tmp_parameters['spherical_wall_center'] = barostat_space['barostat_sphere_center']
                tmp_parameters['spherical_wall_radius'] = self.barostat_parameters['barostat_space_size'][i]
                self.barostat_space_wall_parameters['wall_shape_parameters'].append(tmp_parameters)
            elif self.barostat_parameters['barostat_space_shape'][i].lower() == 'cuboid':
                self.barostat_space_wall_parameters['wall_number'] += 6
                raise Exception('Cuboidal space has not been implemented yet.')
            elif self.barostat_parameters['barostat_space_shape'][i].lower() == 'plane':
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
    def get_volume(space_shape, space_size):
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
            if self.barostat_parameters['barostat_space_shape'][i].lower() == 'sphere':
                self.barostat_space_wall_parameters['wall_shape_parameters'][i]['spherical_wall_radius'] = \
                                                                                self.barostat_parameters['barostat_space_size'][i]
        self.barostat_space_wall.update_wall_parameters(self.barostat_space_wall_parameters)

    def get_barostat_space_bias_forces(self):
        """
        Calculate the cumulative repulsive wall forces acting on atoms in barostat regions.

        Returns:
            np.ndarray: shape (n_atom, 3), total bias forces on each atom from barostat space walls.
        """
        coordinates_all = self.current_system.get_positions()
        barostat_bias_forces = np.zeros((self.n_atom,3))
        for i in range(self.barostat_parameters['barostat_number']):
            index_atom = self.barostat_parameters['barostat_action_atoms'][i]
            coordinates = coordinates_all[index_atom]
            tmp_bias = np.array([self.barostat_space_wall.get_repulsive_wall_force(i_coordinate) for i_coordinate in coordinates])
            for j, j_bias in enumerate(tmp_bias):
                barostat_bias_forces[index_atom[j]] += j_bias
        return barostat_bias_forces

def Berendsen_volume_rescaling(barostat_parameters, time_step, coordinates_all, forces_all, velocities, masses, P_simulation, tau_P):
    '''
    Perform Berendsen volume rescaling for NPT ensemble, defined in Berendsen et al. (1984).
    '''
    Eks = barostat_space.get_atom_kinetic_energies(velocities, masses)
    P_current = np.zeros(len(P_simulation))
    volume = np.zeros(len(P_simulation))
    for i in range(barostat_parameters['barostat_number']):
        index_atom = barostat_parameters['barostat_action_atoms'][i]
        center_of_mass = (coordinates_all[index_atom]*masses[index_atom]).sum(axis=0) / masses[index_atom].sum()
        internal_virial = barostat_space.get_internal_virial(index_atom, coordinates_all, forces_all)
        volume[i] = barostat_space.get_volume(barostat_parameters['barostat_space_shape'][i], barostat_parameters['barostat_space_size'][i])
        P_current[i] = barostat_space.get_pressure(Eks[index_atom], internal_virial, volume[i])
        # factor_mu = 1-time_step*factor_Z/tau_P/3.*(P_simulation[i]-P_current[i])
        # factor_z (compressibility) and tau_p can be combined into single parameter, tau_P,
        # because factor_Z is only used in conjunction with tau_P.
        factor_mu = 1-time_step/tau_P/3.*(P_simulation[i]-P_current[i])
        # It can become unstable if:
        # The time step (time_step) is too large,
        # The pressure relaxation time (tau_P) is too short,
        # The pressure difference is too big.
        # This might result in tmp_miu becoming negative or unreasonably small/large, causing unphysical behavior or numerical divergence.
        # To ensure numerical stability, add clamping:
        # factor_mu = max(0.9, min(1.1, factor_mu))

        if barostat_parameters['barostat_space_type'][i].lower() == 'equilibrium':
            barostat_parameters['barostat_space_size'][i] *= factor_mu
            new_coordinates = (coordinates_all[index_atom] - center_of_mass) * factor_mu + center_of_mass
            coordinates_all[index_atom] = new_coordinates
        elif barostat_parameters['barostat_space_type'][i].lower() == 'ultrafast':
            barostat_parameters['barostat_space_size'][i] *= factor_mu

    return coordinates_all, P_current, volume

def Berendsen_enthalpy(Ek_t, Ep_t, pressure, volume):
    H_enthalpy = np.zeros_like(pressure)
    for i in range(len(pressure)):
        H_enthalpy[i] = Ek_t + Ep_t + pressure[i] * volume[i]
    return H_enthalpy

def Langevin_stage_1_propagate_thermostat(half_time_step, masses, T_simulation, Nf, tau_T, momenta, eta, W_barostat):
    '''
    This function implements Langevin thermostat (Bussi, Zykova-Timan and Parrinello, JCP (2009)) to do isothermal–isobaric ensemble sampling (NPT MD).
    '''
    c = np.exp(-half_time_step / tau_T)

    R_t = np.random.randn(Nf-1).reshape(-1, 3)

    momenta = c*momenta + np.sqrt((1-c**2)*masses/(units.kB*T_simulation)) * R_t
    eta = c*eta + np.sqrt((1-c**2)/(units.kB*T_simulation)/W_barostat) * np.random.randn()

    return momenta, eta

def SVR_stage_1_propagate_thermostat(half_time_step, Ek_t, T_simulation, Nf, tau_T, eta, W_barostat):
    '''
    This function implements stochastic velocity rescaling algorithm (Bussi, Zykova-Timan and Parrinello, JCP (2009)) to do isothermal–isobaric ensenmble sampling (NPT MD).
    '''
    Ek_eta = 0.5 * W_barostat * eta**2
    K_total = Ek_t + Ek_eta

    c = np.exp(-half_time_step / tau_T)
    K_simulation_p = 0.5 * Nf * units.kB * T_simulation
    factor = K_simulation_p / (Nf * K_total)

    S_Nf_1 = np.random.chisquare(Nf - 1)
    R_t = np.random.randn()

    ### alpha
    alpha2 = np.abs(c + (1-c)*(S_Nf_1 + R_t**2)*factor + 2*R_t*np.sqrt(c*(1-c)*factor))
    sign = np.sign(R_t + np.sqrt(c/(1-c)/factor))
    alpha = sign * np.sqrt(alpha2)

    return alpha

def SVR_stage_2_propagate_momenta_eta(half_time_step, eta, momenta, volume, P_current, P_simulation, T_current, W_barostat, forces, masses):
    '''
    This function implements stochastic velocity rescaling algorithm (Bussi, Zykova-Timan and Parrinello, JCP (2009)) to do isothermal–isobaric ensenmble sampling (NPT MD).
    '''
    dot_fp = np.einsum('ij,ij->i', forces, momenta)
    force_sq = np.einsum('ij,ij->i', forces, forces)
    term1 = np.sum(dot_fp / masses)                # ⟨f · p / m⟩
    term2 = np.sum(force_sq / masses)              # ⟨|f|² / m⟩
    # β⁻¹ = k_B T
    prefactor = 3 * (volume * (P_current - P_simulation) + 2 * units.kB * T_current)

    eta_half = eta + prefactor * half_time_step / W_barostat \
               + term1 * half_time_step**2 / W_barostat \
               + term2 * half_time_step**3 / (3 * W_barostat)

    momenta_half = momenta + forces * half_time_step

    return eta_half, momenta_half

def SVR_stage_3_propagate_position_volume(time_step, coordinates, momenta, eta, masses, volume, barostat_space_size, barostat_space_type):
    '''
    This function implements stochastic velocity rescaling algorithm (Bussi, Zykova-Timan and Parrinello, JCP (2009)) to do isothermal–isobaric ensenmble sampling (NPT MD).
    '''
    eta_time_step = eta*time_step
    exp_eta_time_step = np.exp(eta_time_step)
    center_of_mass = (coordinates*masses).sum(axis=0) / masses.sum()

    if barostat_space_type.lower() == 'equilibrium':
        barostat_space_size *= exp_eta_time_step
        coordinates = exp_eta_time_step*(coordinates-center_of_mass) + np.sinh(eta_time_step)/eta * (momenta/masses) + center_of_mass
    elif barostat_space_type.lower() == 'ultrafast':
        barostat_space_size *= np.exp(eta_time_step)

    volume *= np.exp(3 * eta_time_step)

    momenta *= np.exp(-eta_time_step)

    return coordinates, volume, momenta, barostat_space_size

def SVR_effective_enthalpy(Ek_t, Ep_t, eta, volume, T_simulation, pressure, W_barostat):
    """
    Compute effective enthalpy as defined in Eq. (14) of Bussi, Zykova-Timan and Parrinello, JCP (2009).
    """
    # β⁻¹ = k_B T
    beta = 1 / (units.kB * T_simulation)

    H_eff = np.zeros_like(pressure)
    # effective enthalpy
    for i in range(len(pressure)):
        H_eff[i] = Ek_t + Ep_t + pressure[i] * volume[i] - (2 / beta) * np.log(volume[i]) + 0.5 * W_barostat * eta[i]**2
    return H_eff


def SVR_stage_2_4_old(eta, volume, P_current, P_simulation, T_current, half_time_step, W_barostat, forces, momenta, masses, index):
    eta_half = eta + 3*(volume*(P_current-P_simulation)+2*units.kB*T_current)*half_time_step/W_barostat + \
                        np.sum([np.dot(forces[i],momenta[i])/masses[i] for i in range(len(index))])/W_barostat*half_time_step**2 + \
                        np.sum([np.dot(forces[i],forces[i])/masses[i] for i in range(len(index))])/W_barostat/3*half_time_step**3
    momenta_half = momenta + forces * half_time_step
    return eta_half, momenta_half
