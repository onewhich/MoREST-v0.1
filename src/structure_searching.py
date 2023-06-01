import numpy as np
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_potential, on_the_fly, molpro_calculator
from ase import units

class initialize_sampling:
    def __init__(self, morest_parameters, searching_parameters, fire_parameters, calculator=None, log_file=None):
        self.morest_parameters = morest_parameters
        self.searching_parameters = searching_parameters
        self.fire_parameters = fire_parameters
        self.log_file = log_file
        
        if self.morest_parameters['many_body_potential'].upper() in ['on_the_fly'.upper()]:
            if calculator == None:
                raise Exception('Please specify the electronic structure method.')
            self.many_body_potential = on_the_fly(calculator)
        elif self.morest_parameters['many_body_potential'].upper() in ['molpro'.upper()]:
            if type(calculator) == type({}):
                molpro_para_dict = calculator
                self.many_body_potential = molpro_calculator(molpro_para_dict)
            else:
                raise Exception('Please pass the molpro parameters dictionary to calculator.')
        elif self.morest_parameters['many_body_potential'].upper() in ['ML_potential'.upper()]:
            self.ml_calculator = ml_potential(ab_initio_calculator = calculator, \
                                    ml_parameters = self.morest_parameters, \
                                    log_file = self.log_file)
            self.many_body_potential = on_the_fly(self.ml_calculator)
            
        else:
            raise Exception('Which many body potential will you use?')
            
    def get_current_structure(self, molecule=None):
        if self.searching_parameters['searching_initialization']:
            if molecule == None:
                system = read_xyz_file(self.searching_parameters['searching_starting_point'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.log_file.write('Can not read current structure, and read structure from starting point.')
                if molecule == None:
                    system = read_xyz_file(self.searching_parameters['searching_starting_point'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        if self.fire_parameters['fire_equal_masses']:
            self.masses = np.ones(self.n_atom)
        else:
            self.masses = system.get_masses()[:,np.newaxis]

        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        
        return self.current_step, system
    

class FIRE_velocity_Verlet(initialize_sampling):
    '''
    This class implements FIRE structure optimization algorithm based on velocity Verlet integrator.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str (default name) records the initial xyz structure of the system
    MoREST.str_new (default name) records the current xyz structure of the system
    '''
    def __init__(self, morest_parameters, searching_parameters, fire_parameters, calculator=None, log_file=None):
        super(FIRE_velocity_Verlet, self).__init__(morest_parameters, searching_parameters, fire_parameters, calculator, log_file)