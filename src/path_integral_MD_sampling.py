import os
import numpy as np
from copy import deepcopy
from ase import units
from structure_io import read_xyz_traj, write_xyz_file, write_xyz_traj
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from phase_space_sampling import initialize_sampling
from numerical_integraion import RPMD_integration
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, stochastic_velocity_rescaling
from barostat import barostat_space, Berendsen_volume_rescaling, stochastic_velocity_volume_rescaling

class RPMD(initialize_sampling):
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):

        self.beads_traj_file_head = 'MoREST_RPMD_beads_traj_'

        super(RPMD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
        self.beta = RPMD_parameters['beta']
        self.hbar = RPMD_parameters['hbar']
        self.n_beads = RPMD_parameters['rpmd_number_of_beads']
        self.beads_file_name = RPMD_parameters['rpmd_beads_file']
        self.time_step = RPMD_parameters['rpmd_time_step']
        self.T_simulation = RPMD_parameters['rpmd_temperature']
        self.omega_k = RPMD_parameters['omega_k']
        self.C_jk = RPMD_parameters['C_jk']
        self.atom_masses = self.masses.flatten()
        self.current_system.calc = calculator

        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.T_simulation # Ek = 1/2 m v^2 = 3/2 kB T for each particle

        if os.path.isfile(self.beads_file_name):
            self.current_beads = read_xyz_traj(self.beads_file_name)
            if len(self.current_beads) != self.n_beads:
                raise Exception('The number of structures in beads file does not fit the number of beads given by the parameter file. Please check.')
        else:
            self.initialize_beads()
        write_xyz_file(self.beads_file_name, self.current_beads)

        if self.current_step == 0:
            if self.sampling_parameters['sampling_pre_thermalized']:
                if 'sampling_initial_E' in self.sampling_parameters:
                    T_thermalized = 2/3 * self.sampling_parameters['sampling_initial_E']/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_beads[i], temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif 'sampling_initial_T' in self.sampling_parameters:
                    T_thermalized = self.sampling_parameters['sampling_initial_T']
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_beads[i], temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif self.T_simulation > 1e-3:
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_beads[i], temperature_K = self.T_simulation)

        self.current_beads_positions = self.get_beads_positions(self.current_beads)
        self.current_beads_momenta = self.get_beads_momenta(self.current_beads)
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)

        self.update_centroid_positions_momenta(self.current_beads)
        self.update_centroid_potential_energy_forces(self.current_beads_potential_energy, self.current_beads_forces)

        self.integration = RPMD_integration()

    def initialize_beads(self):
        # r_beads: the average distance from a bead to the neighbor for free particles.
        r_beads = [np.sqrt(self.beta * (self.hbar)**2 / self.n_beads / self.atom_masses[i]) for i in range(self.n_atom)] 
        
        # r_ring: the radius of the beads gyration in imaginary time.
        lambda_T = np.array([units._hplanck*units.J*units.s / np.sqrt(2*np.pi*self.atom_masses[i]*units.kB*self.T_simulation) for i in range(self.n_atom)])[:,np.newaxis]
        r_ring = lambda_T / np.sqrt(8*np.pi)

        centroid_pos = self.current_system.get_positions()
        self.current_beads = []

        # the position of the first bead
        # every bead stays r_ring away from the centroid.
        rand_pos_1 = np.random.rand(self.n_atom, 3) - 0.5
        norm_1 = np.linalg.norm(rand_pos_1,axis=-1)[:,np.newaxis]
        pos_new_1 = rand_pos_1/norm_1*r_ring + centroid_pos
        tmp_system = deepcopy(self.current_system)
        tmp_system.set_positions(pos_new_1)
        self.current_beads.append(tmp_system)

        # the positions of other beads
        while len(self.current_beads) < self.n_beads:
            beads_pos = np.array([i_bead.get_positions() for i_bead in self.current_beads])
            tmp_pos = []
            i = 0
            while len(tmp_pos) < self.n_atom:
                atoms_pos = beads_pos[:,i,:]

                rand_pos = np.random.rand(3) - 0.5
                norm = np.linalg.norm(rand_pos,axis=-1)
                pos_new = np.array(rand_pos/norm*r_ring[i] + centroid_pos[i])

                # each bead is farther from every other bead than r_beads.
                if np.all(np.linalg.norm(atoms_pos - pos_new,axis=-1) > r_beads[i]):
                    tmp_pos.append(pos_new)
                    i += 1
            tmp_system = deepcopy(self.current_system)
            tmp_system.set_positions(tmp_pos)
            self.current_beads.append(tmp_system)

    def RPMD_update_step(self, next_beads_momenta, next_beads_positions):
        self.update_beads_positions(next_beads_positions)
        self.current_beads_positions = next_beads_positions
        self.update_beads_momenta(next_beads_momenta)
        self.current_beads_momenta = next_beads_momenta
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)
        write_xyz_file(self.beads_file_name, self.current_beads)
        self.current_step += 1
        self.update_centroid_positions_momenta(self.current_beads)
        self.update_centroid_potential_energy_forces(self.current_beads_potential_energy, self.current_beads_forces)
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

    def pre_thermalization(self, Tf):
        for i in range(self.n_beads):
            Ek_i = self.current_beads[i].get_kinetic_energy()
            Ti = 2/3 * Ek_i/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
            velocities = self.current_beads[i].get_velocities()
            factor = np.sqrt(Tf / Ti)
            self.current_beads[i].set_velocities(factor * velocities)
            
    def get_beads_positions(self, beads):
        beads_positions = np.array([i_bead.get_positions() for i_bead in beads])
        return beads_positions
    
    def update_beads_positions(self, new_beads_positions):
        for i in range(self.n_beads):
            self.current_beads[i].set_positions(new_beads_positions[i])

    def get_beads_momenta(self, beads):
        beads_momenta = np.array([i_bead.get_momenta() for i_bead in beads])
        return beads_momenta
    
    def update_beads_momenta(self, new_beads_momenta):
        for i in range(self.n_beads):
            self.current_beads[i].set_momenta(new_beads_momenta[i])
            
    def get_beads_potential_forces(self, beads):
        beads_potential_energy = []
        beads_forces = []
        for i in range(self.n_beads):
            tmp_potential_energy, tmp_forces = self.many_body_potential.get_potential_forces(beads[i])
            beads_potential_energy.append(tmp_potential_energy)
            beads_forces.append(tmp_forces)
        return np.array(beads_potential_energy), np.array(beads_forces)
    
    def update_centroid_positions_momenta(self, beads):
        beads_positions = self.get_beads_positions(beads)
        system_positions = np.average(beads_positions, axis=0)
        self.current_system.set_positions(system_positions)
        beads_momenta = self.get_beads_momenta(beads)
        system_momenta = np.average(beads_momenta, axis=0)
        self.current_system.set_momenta(system_momenta)

    def update_centroid_potential_energy_forces(self, beads_potential_energy, beads_forces):
        self.current_system.calc.results['energy'] = np.average(beads_potential_energy)
        self.current_system.calc.results['forces'] = np.average(beads_forces, axis=0)

    # only remove the centroid translational motion
    def stationary_centroid(self):
        centroid_velocity = self.current_system.get_velocities()
        new_velocity = self.clean_translation(centroid_velocity)
        self.current_system.set_velocities(new_velocity)
        d_velocity = new_velocity - centroid_velocity
        for i_bead in self.current_beads:
            bead_velocity = i_bead.get_velocities()
            i_bead.set_velocities(bead_velocity + d_velocity)

    # only remove the centroid rotational motion
    def zero_rotation_centroid(self):
        centroid_velocity = self.current_system.get_velocities()
        ZeroRotation(self.current_system)
        new_velocity = self.current_system.get_velocities()
        d_velocity = new_velocity - centroid_velocity
        for i_bead in self.current_beads:
            bead_velocity = i_bead.get_velocities()
            i_bead.set_velocities(bead_velocity + d_velocity)

    @staticmethod
    def clean_translation(velocities):
        total_velocity = np.sum(velocities, axis=0)/len(velocities)
        new_velocities = velocities - total_velocity
        return new_velocities
        
    @staticmethod
    def clean_rotation(velocities, coordinates, masses):
        '''
        L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
        L : angular momentum
        omega: angular velocity
        I : moment of inertia
        '''
        v_vector = velocities
        #center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
        center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
        r_vector = coordinates - center_of_mass
        # r_cross_v : angular velocities
        # omega = (r x v) / |r|^2
        r_cross_v = np.cross(r_vector, v_vector)
        r_2 = np.linalg.norm(r_vector, axis=1)**2
        omega = np.array([r_cross_v[i]/r_2[i] for i in range(4)])
        # Rv = omega/n_atom : system total angular velocity
        rotat_vector = np.sum(omega, axis=0)/len(masses)
        v_tang = np.cross(rotat_vector, r_vector)
        new_velocities = v_vector - v_tang
        return new_velocities
        
    @staticmethod
    def write_RPMD_log(RPMD_log, step, Ep, Ek, masses):
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
        except:
            pass
        n_atom = len(masses)
        T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        Et = Ek + Ep
        RPMD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')

    @staticmethod
    def write_SVR_RPMD_log(RPMD_log, step, Ep, Ek, masses, Ee=0, d_Ee=0):
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
        except:
            pass
        n_atom = len(masses)
        #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
        #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
        T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        Et = Ek + Ep
        Ee += d_Ee
        RPMD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
        return Ee

class RP_NVE(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None, time_step=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        if type(time_step) == type(None):
            time_step = self.time_step

        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.integration.propagate_momenta_half(time_step, self.current_beads_momenta, self.current_beads_forces)
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k, beads_positions_k = self.integration.transform_to_normal_mode(beads_momenta_half, self.current_beads_positions, \
                                                                                            self.C_jk, self.n_atom, self.n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp, beads_positions_kp = self.integration.free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, \
                                                                                          self.omega_k, self.n_atom, self.n_beads, self.atom_masses)
        # back transform momenta and positions
        beads_momenta_half, next_beads_positions = self.integration.transform_back_to_coordinates(beads_momenta_half_kp, beads_positions_kp, \
                                                                                                  self.C_jk, self.n_atom, self.n_beads)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = self.integration.propagate_momenta_half(time_step, beads_momenta_half, self.current_beads_forces)

        self.RPMD_update_step(next_beads_momenta, next_beads_positions)

        if self.RPMD_clean_translation:
            self.stationary_centroid()
        if self.RPMD_clean_rotation:
            self.zero_rotation_centroid()
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVK_VR(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, old_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        self.update_centroid_positions_momenta(self.current_beads)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None, time_step=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        if type(time_step) == type(None):
            time_step = self.time_step

        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.integration.propagate_momenta_half(time_step, self.current_beads_momenta, self.current_beads_forces)
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k, beads_positions_k = self.integration.transform_to_normal_mode(beads_momenta_half, self.current_beads_positions, \
                                                                                            self.C_jk, self.n_atom, self.n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp, beads_positions_kp = self.integration.free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, \
                                                                                          self.omega_k, self.n_atom, self.n_beads, self.atom_masses)
        # back transform momenta and positions
        beads_momenta_half, next_beads_positions = self.integration.transform_back_to_coordinates(beads_momenta_half_kp, beads_positions_kp, \
                                                                                                  self.C_jk, self.n_atom, self.n_beads)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = self.integration.propagate_momenta_half(time_step, beads_momenta_half, self.current_beads_forces)

        self.RPMD_update_step(next_beads_momenta, next_beads_positions)

        if self.RPMD_clean_translation:
            self.stationary_centroid()
        if self.RPMD_clean_rotation:
            self.zero_rotation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVT_Berendsen(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], old_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        self.update_centroid_positions_momenta(self.current_beads)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None, time_step=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        if type(time_step) == type(None):
            time_step = self.time_step

        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.integration.propagate_momenta_half(time_step, self.current_beads_momenta, self.current_beads_forces)
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k, beads_positions_k = self.integration.transform_to_normal_mode(beads_momenta_half, self.current_beads_positions, \
                                                                                            self.C_jk, self.n_atom, self.n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp, beads_positions_kp = self.integration.free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, \
                                                                                          self.omega_k, self.n_atom, self.n_beads, self.atom_masses)
        # back transform momenta and positions
        beads_momenta_half, next_beads_positions = self.integration.transform_back_to_coordinates(beads_momenta_half_kp, beads_positions_kp, \
                                                                                                  self.C_jk, self.n_atom, self.n_beads)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = self.integration.propagate_momenta_half(time_step, beads_momenta_half, self.current_beads_forces)

        self.RPMD_update_step(next_beads_momenta, next_beads_positions)

        if self.RPMD_clean_translation:
            self.stationary_centroid()
        if self.RPMD_clean_rotation:
            self.zero_rotation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVT_Langevin(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            self.Ee = self.write_SVR_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None, time_step=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        if type(time_step) == type(None):
            time_step = self.time_step

        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.integration.propagate_momenta_half(time_step, self.current_beads_momenta, self.current_beads_forces)
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k, beads_positions_k = self.integration.transform_to_normal_mode(beads_momenta_half, self.current_beads_positions, \
                                                                                            self.C_jk, self.n_atom, self.n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp, beads_positions_kp = self.integration.free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, \
                                                                                          self.omega_k, self.n_atom, self.n_beads, self.atom_masses)
        # back transform momenta and positions
        beads_momenta_half, next_beads_positions = self.integration.transform_back_to_coordinates(beads_momenta_half_kp, beads_positions_kp, \
                                                                                                  self.C_jk, self.n_atom, self.n_beads)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = self.integration.propagate_momenta_half(time_step, beads_momenta_half, self.current_beads_forces)

        self.RPMD_update_step(next_beads_momenta, next_beads_positions)

        if self.RPMD_clean_translation:
            self.stationary_centroid()
        if self.RPMD_clean_rotation:
            self.zero_rotation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  1, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.Ee = self.write_SVR_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system
    
class RP_NVT_SVR(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            self.Ee = self.write_SVR_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None, time_step=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        if type(time_step) == type(None):
            time_step = self.time_step

        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.integration.propagate_momenta_half(time_step, self.current_beads_momenta, self.current_beads_forces)
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k, beads_positions_k = self.integration.transform_to_normal_mode(beads_momenta_half, self.current_beads_positions, \
                                                                                            self.C_jk, self.n_atom, self.n_beads)
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp, beads_positions_kp = self.integration.free_beads_evolution(time_step, beads_positions_k, beads_momenta_half_k, \
                                                                                          self.omega_k, self.n_atom, self.n_beads, self.atom_masses)
        # back transform momenta and positions
        beads_momenta_half, next_beads_positions = self.integration.transform_back_to_coordinates(beads_momenta_half_kp, beads_positions_kp, \
                                                                                                  self.C_jk, self.n_atom, self.n_beads)
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = self.integration.propagate_momenta_half(time_step, beads_momenta_half, self.current_beads_forces)

        self.RPMD_update_step(next_beads_momenta, next_beads_positions)

        if self.RPMD_clean_translation:
            self.stationary_centroid()
        if self.RPMD_clean_rotation:
            self.zero_rotation_centroid()

        # only rescale the centroids 
        old_velocities = self.current_system.get_velocities()
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.Ee = self.write_SVR_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system