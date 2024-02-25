import os
import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_file, write_xyz_traj
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from phase_space_sampling import initialize_sampling

class rpmd(initialize_sampling):
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, morest_parameters, sampling_parameters, rpmd_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):
        self.n_beads = rpmd_parameters['rpmd_number_of_beads']
        self.temperature = rpmd_parameters['rpmd_temperature']
        self.beads_file_name = rpmd_parameters['rpmd_beads_file']

        if os.path.isfile(self.beads_file_name):
            beads = read_xyz_traj(self.beads_file_name)
        else:
            super(rpmd, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)