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
        Eks = self.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        all_forces = self.current_system.get_forces()
        all_coordinates = self.current_system.get_positions()
        for i in range(self.barostat_parameters['barostat_number']):
            index = self.barostat_parameters['barostat_action_atoms'][i]
            if index == 'all':
                index = np.arange(self.n_atom)
                self.barostat_parameters['barostat_action_atoms'][i] = index
            internal_virial = self.get_internal_virial(index, all_coordinates, all_forces)
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
        v = np.linalg.norm(velocities, axis=-1)[:,np.newaxis]
        Eks = 0.5 * masses * v**2
        return Eks

    @staticmethod
    def get_internal_virial(index, all_coordinates, all_forces):
        index_all = np.arange(len(all_coordinates))
        index_exclude = np.setdiff1d(index_all, index)
        coordinates = all_coordinates[index]
        forces = all_forces[index]
        n_atom = len(index)
        # virial belonged to the specified barostat
        inner_virial = -np.sum([(coordinates[i]-coordinates[j]) @ forces[i] for i in range(n_atom-1) for j in range(i+1,n_atom)])/2
        # virial belonged to the interaction between specified barostat and other barostat
        outer_virial = -np.sum([(all_coordinates[i]-all_coordinates[j]) @ all_forces[i] for i in index for j in index_exclude])/2
        outer_virial /= 2
        return inner_virial+outer_virial

    @staticmethod
    def get_volume(space_shape, space_size):
        if space_shape.upper() == 'sphere'.upper():
            volume =  4./3. * np.pi * space_size**3 # V = 4/3 * Pi * r^3
        elif space_shape.upper() == 'cuboid'.upper():
            raise Exception('Cuboid space has not been implemented yet.')
        elif space_shape.upper() == 'plane'.upper():
            raise Exception('Planar space has not been implemented yet.')
        return volume
    
    @staticmethod
    def get_pressure(Ek_atoms, internal_virial, volume):
        return (np.sum(Ek_atoms) - internal_virial)*2/(3 * volume)

    def update_barostat_space_wall(self):
        for i in range(self.barostat_parameters['barostat_number']):
            if self.barostat_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_space_wall_parameters['wall_shape_parameters'][i]['spherical_wall_radius'] = \
                                                                                self.barostat_parameters['barostat_space_size'][i]
        self.barostat_space_wall.update_wall_parameters(self.barostat_space_wall_parameters)

    def get_barostat_space_bias_forces(self):
        all_coordinates = self.current_system.get_positions()
        barostat_bias_forces = np.zeros((self.n_atom,3))
        for i in range(self.barostat_parameters['barostat_number']):
            index = self.barostat_parameters['barostat_action_atoms'][i]
            coordinates = all_coordinates[index]
            tmp_bias = np.array([self.barostat_space_wall.get_repulsive_wall_force(i_coordinate) for i_coordinate in coordinates])
            for j, j_bias in enumerate(tmp_bias):
                barostat_bias_forces[index[j]] += j_bias
        return barostat_bias_forces

def Berendsen_volume_rescaling(barostat_parameters, time_step, all_coordinates, all_forces, velocities, masses, P_simulation, tau_P, beta, factor_miu):
    Eks = barostat_space.get_atom_kinetic_energies(velocities, masses)
    next_miu = []
    P_current = []
    for i in range(barostat_parameters['barostat_number']):
        index = barostat_parameters['barostat_action_atoms'][i]
        internal_virial = barostat_space.get_internal_virial(index, all_coordinates, all_forces)
        volume = barostat_space.get_volume(barostat_parameters['barostat_space_shape'][i], barostat_parameters['barostat_space_size'][i])
        P_current.append(barostat_space.get_pressure(Eks[index], internal_virial, volume))
        tmp_miu = np.power(1+(time_step/tau_P)*beta*(P_current-P_simulation[i]),1./3.)
        next_miu.append(tmp_miu)

        if barostat_parameters['barostat_space_type'][i].upper() == 'gas'.upper():
            barostat_parameters['barostat_space_size'][i] *= factor_miu[i]
        elif barostat_parameters['barostat_space_type'][i].upper() == 'condensed'.upper():
            barostat_parameters['barostat_space_size'][i] *= factor_miu[i]
            all_coordinates[index] *= factor_miu[i]

    return all_coordinates, np.array(next_miu), np.array(P_current)
    
def stochastic_velocity_volume_rescaling(barostat_parameters, time_step, half_time_step, all_coordinates, all_forces, velocities, eta, momenta, \
                                         masses, W_barostat, T_current, P_simulation):
    Eks = barostat_space.get_atom_kinetic_energies(velocities, masses)
    P_current = []
    for i in range(barostat_parameters['barostat_number']):
        index = barostat_parameters['barostat_action_atoms'][i]

        # stage 2
        internal_virial = barostat_space.get_internal_virial(index, all_coordinates, all_forces)
        volume = barostat_space.get_volume(barostat_parameters['barostat_space_shape'][i], barostat_parameters['barostat_space_size'][i])
        P_current.append(barostat_space.get_pressure(Eks[index], internal_virial, volume))
        eta_half, momenta_half = SVR_stage_2_4(eta[i], volume, P_current[i], P_simulation[i], T_current, half_time_step, W_barostat, \
                                               all_forces[index], momenta[index], masses[index], index)
        # stage 3
        eta_half_time_step = eta_half*time_step
        volume *= np.exp(3*eta_half_time_step)
        momenta_half *= np.exp(-1*eta_half_time_step)
        if barostat_parameters['barostat_space_type'][i].upper() == 'gas'.upper():
            barostat_parameters['barostat_space_size'][i] *= np.exp(eta_half_time_step)
        elif barostat_parameters['barostat_space_type'][i].upper() == 'condensed'.upper():
            barostat_parameters['barostat_space_size'][i] *= np.exp(eta_half_time_step)
            all_coordinates[index] = np.exp(eta_half_time_step)*all_coordinates[index] + \
                                     np.sinh(eta_half_time_step)/eta_half * \
                                     (momenta_half/masses[index])
        # stage 4
        internal_virial = barostat_space.get_internal_virial(index, all_coordinates, all_forces)
        P_current[i] = barostat_space.get_pressure(0.5*momenta_half**2/masses[index], internal_virial, volume)
        new_eta, new_momenta = SVR_stage_2_4(eta_half, volume, P_current[i], P_simulation[i], T_current, half_time_step, W_barostat, \
                                               all_forces[index], momenta_half, masses[index], index)

        eta[i] = new_eta
        momenta[index] = new_momenta

    return all_coordinates, momenta, eta, P_current

def SVR_stage_2_4(eta, volume, P_current, P_simulation, T_current, half_time_step, W_barostat, forces, momenta, masses, index):
    eta_half = eta + 3*(volume*(P_current-P_simulation)+2*units.kB*T_current)*half_time_step/W_barostat + \
                        np.sum([np.dot(forces[i],momenta[i])/masses[i] for i in range(len(index))])/W_barostat*half_time_step**2 + \
                        np.sum([np.dot(forces[i],forces[i])/masses[i] for i in range(len(index))])/W_barostat/3*half_time_step**3
    momenta_half = momenta + forces * half_time_step
    return eta_half, momenta_half
