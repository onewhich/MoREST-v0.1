from time import time
import os
import numpy as np
from copy import deepcopy
from ase import units
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_file, write_xyz_traj
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from phase_space_sampling import initialize_sampling

class RPMD(initialize_sampling):
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):

        self.beads_traj_file_head = 'MoREST_RPMD_beads_traj_'

        time_0 = time()
        super(RPMD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
        time_1 = time()
        print('time intialize sampling:', time_1-time_0)
        self.n_beads = RPMD_parameters['rpmd_number_of_beads']
        self.beads_file_name = RPMD_parameters['rpmd_beads_file']
        self.time_step = RPMD_parameters['rpmd_time_step']
        self.temperature = RPMD_parameters['rpmd_temperature']
        self.omega_k = RPMD_parameters['omega_k']
        self.C_jk = RPMD_parameters['C_jk']
        self.atom_masses = self.masses.flatten()

        if self.current_step == 0:
            if self.sampling_parameters['sampling_pre_thermalized']:
                if 'sampling_initial_E' in self.sampling_parameters:
                    T_thermalized = 2/3 * self.sampling_parameters['sampling_initial_E']/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif 'sampling_initial_T' in self.sampling_parameters:
                    T_thermalized = self.sampling_parameters['sampling_initial_T']
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif self.T_simulation > 1e-3:
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)

        time_0 = time()
        if os.path.isfile(self.beads_file_name):
            self.current_beads = read_xyz_traj(self.beads_file_name)
            if len(self.current_beads) != self.n_beads:
                raise Exception('The number of structures in beads file does not fit the number of beads given by the parameter file. Please check.')
        else:
            self.current_beads = []
            self.current_beads.append(deepcopy(self.current_system))
            for _ in range(self.n_beads-1):
                self.VV_initialize_beads()
                self.current_beads.append(deepcopy(self.current_system))
        write_xyz_file(self.beads_file_name, self.current_beads)
        time_1 = time()
        print('time prepare beads:', time_1-time_0)

        time_0 = time()
        self.current_beads_positions = self.get_beads_positions(self.current_beads)
        time_1 = time()
        print('time get positions:', time_1-time_0)
        time_0 = time()
        self.current_beads_momenta = self.get_beads_momenta(self.current_beads)
        time_1 = time()
        print('time get momenta:', time_1-time_0)
        time_0 = time()
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)
        time_1 = time()
        print('time get energy and forces:', time_1-time_0)

        self.update_current_system_from_beads_average(self.current_beads_positions, self.current_beads_momenta)

    def VV_initialize_beads(self, time_step=None, bias_forces=None, temperature=None):
        if type(time_step) == type(None):
            time_step = self.time_step

        if type(temperature) == type(None):
            temperature = 1000
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            self.current_forces = self.current_forces + bias_forces

        MaxwellBoltzmannDistribution(self.current_system, temperature_K = temperature)
        
        ### x(t), v(t) = p(t) / m
        current_coordinates = self.current_system.get_positions()
        #current_velocities = self.current_system.get_velocities()
        current_momenta = self.current_system.get_momenta()
        
        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        #next_coordinates = current_coordinates + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_coordinates = current_coordinates + (current_momenta * time_step + 0.5 * self.current_forces * time_step**2) / self.masses
        self.current_system.set_positions(next_coordinates)
        
        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * self.current_forces * time_step
        
        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * self.time_step
        #self.current_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        self.current_system.set_momenta(next_momenta)
        
        #next_velocities = next_system.get_velocities()
    
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy

    def RPMD_next_step(self, time_step=None, wall_potential=None, updated_current_beads=None):
        if type(time_step) == type(None):
            time_step = self.time_step

        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(wall_potential) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                current_positions = self.current_beads_positions[i]
                bias_force = wall_potential(current_positions)
                self.current_beads_forces[i] = current_forces + bias_force
            
        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.current_beads_momenta + 0.5 * time_step * self.current_beads_forces
        
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k = self.coordinate_to_normal_mode_representation(beads_momenta_half)
        beads_positions_k = self.coordinate_to_normal_mode_representation(self.current_beads_positions)
            
        # dt Hamiltonian kinetic energy part
        beads_momenta_half_kp = np.array([[np.cos(self.omega_k[k]*self.time_step)*beads_momenta_half_k[k,i,:] for i in range(self.n_atom)] \
                                    for k in range(self.n_beads)]) \
                                - np.array([[self.atom_masses[i]*self.omega_k[k]*np.sin(self.omega_k[k]*self.time_step)*beads_positions_k[k,i,:] \
                                    for i in range(self.n_atom)] for k in range(self.n_beads)])
        beads_positions_kp = np.array([[1/(self.atom_masses[i]*self.omega_k[k])*np.sin(self.omega_k[k]*self.time_step)*beads_momenta_half_k[k,i,:] \
                               for i in range(self.n_atom)] for k in range(self.n_beads)]) \
                             + np.array([[np.cos(self.omega_k[k]*self.time_step)*beads_positions_k[k,i,:] for i in range(self.n_atom)] \
                                for k in range(self.n_beads)])

        # back transform momenta and positions
        beads_momenta_half = self.normal_mode_to_coordinate_representation(beads_momenta_half_kp)
        next_beads_positions = self.normal_mode_to_coordinate_representation(beads_positions_kp)
            
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = beads_momenta_half + 0.5 * time_step * self.current_beads_forces

        self.update_beads_positions(next_beads_positions)
        self.current_beads_positions = next_beads_positions
        self.update_beads_momenta(next_beads_momenta)
        self.current_beads_momenta = next_beads_momenta
        write_xyz_file(self.beads_file_name, self.current_beads)
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)
        write_xyz_file(self.beads_file_name, self.current_beads)
        self.current_step += 1
        self.update_current_system_from_beads_average(self.current_beads_positions, self.current_beads_momenta)
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

    def pre_thermalization(self, Tf):
        Ek_i = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek_i/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        velocities = self.current_system.get_velocities()
        factor = np.sqrt(Tf / Ti)
        self.current_system.set_velocities(factor * velocities)
            
    def get_beads_positions(self, beads):
        beads_positions = np.array([beads[i].get_positions() for i in range(self.n_beads)])
        return beads_positions
    
    def update_beads_positions(self, new_beads_positions):
        for i in range(self.n_beads):
            self.current_beads[i].set_positions(new_beads_positions[i])

    def get_beads_momenta(self, beads):
        beads_momenta = np.array([beads[i].get_momenta() for i in range(self.n_beads)])
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
    
    def coordinate_to_normal_mode_representation(self, beads_vectors):
        return np.array([[np.sum([beads_vectors[j,i,:]*self.C_jk[j,k] for j in range(self.n_beads)],axis=0) \
                          for i in range(self.n_atom)] for k in range(self.n_beads)])

    def normal_mode_to_coordinate_representation(self, beads_vectors):
        return np.array([[np.sum([beads_vectors[j,i,:]*self.C_jk[k,j] for j in range(self.n_beads)],axis=0) \
                          for i in range(self.n_atom)] for k in range(self.n_beads)])
    
    def update_current_system_from_beads_average(self, beads_positions, beads_momenta):
        system_positions = np.average(beads_positions, axis=0)
        self.current_system.set_positions(system_positions)
        system_momenta = np.average(beads_momenta, axis=0)
        self.current_system.set_momenta(system_momenta)
        
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

class RP_NVE(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD_NVE.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_NVE_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        time_0 = time()
        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)
        time_1 = time()
        print('time write the log:', time_1-time_0)

    def generate_new_step(self, wall_potential=None, updated_current_beads=None):
        self.RPMD_next_step(wall_potential=wall_potential, updated_current_beads=updated_current_beads)

        if self.RPMD_clean_translation:
            Stationary(self.current_beads[0])
        if self.RPMD_clean_rotation:
            ZeroRotation(self.current_beads[0])
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_traj_file_head+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_RPMD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    