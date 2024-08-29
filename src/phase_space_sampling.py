import numpy as np
from structure_io import read_xyz_file, read_xyz_traj
from initialize_calculator import initialize_calculator

class initialize_sampling(initialize_calculator):
    def __init__(self, morest_parameters, sampling_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):
        super(initialize_sampling, self).__init__(morest_parameters, calculator, log_morest)
        self.sampling_parameters = sampling_parameters

        if self.sampling_parameters['sampling_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
        else:
            try:
                self.current_traj = read_xyz_traj(traj_file_name)
                self.current_step = (len(self.current_traj) - 1) * self.sampling_parameters['sampling_traj_interval']
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.current_step = 0
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure(molecule)
            
    def get_current_structure(self, molecule=None):
        if self.sampling_parameters['sampling_initialization']:
            if type(molecule) == type(None):
                system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
                if type(molecule) == type(None):
                    system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses

        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return system
            