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
        self.n_beads = RPMD_parameters['rpmd_number_of_beads']
        self.beads_file_name = RPMD_parameters['rpmd_beads_file']
        self.time_step = RPMD_parameters['rpmd_time_step']
        self.temperature = RPMD_parameters['rpmd_temperature']
        self.omega_k = RPMD_parameters['omega_k']
        self.C_jk = RPMD_parameters['C_jk']

        if os.path.isfile(self.beads_file_name):
            beads = read_xyz_traj(self.beads_file_name)
            if len(beads) != self.n_beads:
                raise Exception('The number of structures in beads file does not fit the number of beads given by the parameter file. Please check.')
        else:
            beads = []
            super(RPMD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
            for _ in range(self.n_beads):
                beads.append(deepcopy(self.current_system))

    def RPMD_next_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        if type(time_step) == type(None):
            time_step = self.time_step

        if type(updated_current_system) != type(None):
            self.current_system = updated_current_system
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            self.current_forces = self.current_forces + bias_forces