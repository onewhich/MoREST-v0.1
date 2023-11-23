import numpy as np
from ase import units
from wall_potential import repulsive_wall

class barostat_space:
    def __init__(self, md_parameters, current_system):
        self.md_parameters = md_parameters
        self.current_system = current_system
        self.n_atom = current_system.get_global_number_of_atoms()
        self.masses = current_system.self.get_masses()
        self.P_simulation = md_parameters['barostat_pressure']
        self.initialize_barostat_space_size()
        self.initialize_barostat_space_wall()

    def initialize_barostat_space_size(self):
        Eks = self.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        all_forces = self.current_system.get_forces()
        all_coordinates = self.current_system.get_positions()
        for i in range(self.md_parameters['barostat_number']):
            index = self.md_parameters['barostat_action_atoms'][i]
            if index == 'all':
                index = np.arange(self.n_atom)
                self.md_parameters['barostat_action_atoms'][i] = index
            internal_virial = self.get_internal_virial(index, all_coordinates, all_forces)
            volume = 2*(Eks[i] - internal_virial)/(3*self.P_simulation[i])
            if self.md_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.md_parameters['barostat_space_size'].append(np.pow((3/4 * volume / np.pi), 1./3.))  # V = 4/3 * Pi * r^3; r = (3/4 * V/Pi)^(1/3)

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
        for i, barostat_space in enumerate(self.md_parameters['barostat_space_parameters']):
            if self.md_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_space_wall_parameters['wall_number'] += 1
                self.barostat_space_wall_parameters['wall_collective_variable'].append(self.md_parameters['barostat_collective_variable'][i])
                self.barostat_space_wall_parameters['wall_shape'].append('spherical')
                self.barostat_space_wall_parameters['wall_type'].append('power_wall')
                self.barostat_space_wall_parameters['power_wall_direction'].append(-1)
                self.barostat_space_wall_parameters['wall_scaling'].append(1)
                self.barostat_space_wall_parameters['wall_scope'].append(2)
                self.barostat_space_wall_parameters['wall_action_atoms'].append(self.md_parameters['barostat_number'][i])
                tmp_parameters = {}
                tmp_parameters['spherical_wall_center'] = barostat_space['barostat_sphere_center']
                tmp_parameters['spherical_wall_radius'] = self.md_parameters['barostat_space_size'][i]
                self.barostat_space_wall_parameters['wall_shape_parameters'].append(tmp_parameters)
            elif self.md_parameters['barostat_space_shape'][i].upper() == 'cuboid'.upper():
                self.barostat_space_wall_parameters['wall_number'] += 6
                raise Exception('Cuboidal space has not been implemented yet.')
            elif self.md_parameters['barostat_space_shape'][i].upper() == 'plane'.upper():
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

    def update_barostat_space_wall(self):
        for i in range(self.md_parameters['barostat_number']):
            if self.md_parameters['barostat_space_shape'][i].upper() == 'sphere'.upper():
                self.barostat_space_wall_parameters['wall_shape_parameters'][i]['spherical_wall_radius'] = self.md_parameters['barostat_space_size'][i]
        self.barostat_space_wall.update_wall_parameters(self.barostat_space_wall_parameters)

    def get_barostat_space_bias_forces(self):
        all_coordinates = self.current_system.get_positions()
        barostat_bias_forces = np.zeros((self.n_atom,3))
        for i in range(self.md_parameters['barostat_number']):
            index = self.md_parameters['barostat_action_atoms'][i]
            coordinates = all_coordinates[index]
            tmp_bias = np.array([self.barostat_space_wall.get_repulsive_wall_force(i_coordinate) for i_coordinate in coordinates])
            for j, j_bias in enumerate(tmp_bias):
                barostat_bias_forces[index[j]] += j_bias
        return barostat_bias_forces

def Berendsen_volume_rescaling(md_parameters, forces, coordinates, P_simulation, tau_P, beta, factor):
    time_step = md_parameters['md_time_step']
    Eks = barostat_space.get_atom_kinetic_energies()
    next_factor = []
    for i in range(md_parameters['barostat_number']):
        index = md_parameters['barostat_action_atoms'][i]
        internal_virial = barostat_space.get_internal_virial(coordinates[index], forces[index])
        V = barostat_space.get_volume(md_parameters['barostat_space_shape'][i], md_parameters['barostat_space_size'][i])
        current_pressure = (Eks[i] - internal_virial)*2/(3 * V)
        tmp_factor = np.power(1+(time_step/tau_P)*beta*(current_pressure-P_simulation[i]),1./3.)
        next_factor.append(tmp_factor)

        if md_parameters['barostat_space_type'][i].upper() == 'gas'.upper():
            md_parameters['barostat_space_size'][i] *= factor[i]
        elif md_parameters['barostat_space_type'][i].upper() == 'condensed'.upper():
            md_parameters['barostat_space_size'][i] *= factor[i]
            coordinates[index] *= factor[i]

    return coordinates, np.array(next_factor)
    