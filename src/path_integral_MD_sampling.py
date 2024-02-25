import numpy as np
from structure_io import write_xyz_file, write_xyz_traj
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from phase_space_sampling import initialize_sampling

class rpmd(initialize_sampling):
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, rp_parameters, beads_file):
        self.n_beads = rp_parameters['rp_number_of_beads']
        