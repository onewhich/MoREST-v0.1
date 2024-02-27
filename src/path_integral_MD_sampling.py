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
            super(RPMD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
            for _ in range(self.n_beads):
                self.current_beads.append(deepcopy(self.current_system))
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

    def RPMD_next_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(time_step) == type(None):
            time_step = self.time_step

        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        #if type(bias_forces) != type(None):
        #    self.current_forces = self.current_forces + bias_forces
            
        # p_j(t+0.5dt) = p_j(t) + 0.5 * dt * F(t)
        beads_momenta_half = self.current_beads_momenta + 0.5 * time_step * self.current_beads_forces
        
        # transform momenta and positions from coordinate representation to normal mode representation
        beads_momenta_half_k : beads_momenta_half
        beads_positions_k : self.current_beads_positions
            
        # dt Hamiltonian kinetic energy part
            

        # back transform momenta and positions
        beads_momenta_half : beads_momenta_half_k
        next_beads_positions : beads_positions_k
            
        # p_j(t+dt) = p_j(t+0.5dt) + 0.5 * dt * F(t)
        next_beads_momenta = beads_momenta_half + 0.5 * time_step * self.current_beads_forces

        self.update_beads_positions(next_beads_positions)
        self.update_beads_momenta(next_beads_momenta)
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)
        self.current_step += 1
            
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
        beads_momenta = np.array([beads[i].get_mometa() for i in range(self.n_beads)])
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
        