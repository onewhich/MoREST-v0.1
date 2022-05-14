import numpy as np
#import sys
#sys.path.append('..')
from structure import read_xyz_file, write_xyz_file
from many_body_potential import ml_potential

class velocity_Verlet:
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling.
    MoREST.traj records the trajectory in an extended xyz format:
    ------------------------------------
    n_atoms
    current_step    total_energy
    element1    x1    y1    z1    velocity_x1    velocity_y1    velocity_z1    force_x1    force_y1    force_z1
    ...
    ------------------------------------
    MoREST.str records the initial xyz structure of the system
    MoREST.str_new records the current xyz structure of the system
    '''
    
    def __init__(self, sampling_parameters, md_parameters):
        #self.md_parameters = np.load('MoREST_md_parameters.npy',allow_pickle=True).item()
        self.sampling_parameters = sampling_parameters
        self.md_parameters = md_parameters
        
        self.structure = self.get_current_structure()
        
        if self.sampling_parameters['sampling_restart']:
            self.__log_traj = open('MoREST.traj','a')
        else:
            self.__log_traj = open('MoREST.traj','w')
            write_xyz_file(self.__log_traj, self.structure)
        
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.md_parameters['md_time_step']
        
        self.next_structure = {}
        self.next_structure['n_atoms'] = self.structure['n_atoms']
        self.next_structure['elements'] = self.structure['elements']
        self.next_structure['current_step'] = self.structure['current_step'] + 1
        self.next_structure['masses'] = self.structure['masses']
        self.next_structure['coordinates'] = self.structure['coordinates'] \
                                           + self.structure['velocities'] * time_step \
                                           + 0.5 * self.structure['accelerations'] * time_step**2
        self.next_structure['total_energy'], self.next_structure['forces'] = \
                                           ml_potential().get_potential_FD_forces(self.next_structure['coordinates'])
        if bias_forces != None:
            self.next_structure['forces'] = self.next_structure['forces'] + bias_forces        
        self.next_structure['accelerations'] = np.array([self.next_structure['forces'][i_atom]/self.next_structure['masses'][i_atom] \
                                              for i_atom in range(len(self.next_structure['forces']))])
        self.next_structure['velocities'] = self.structure['velocities'] \
                                          + 0.5 * (self.structure['accelerations'] + self.next_structure['accelerations']) * time_step
        
        self.structure = self.next_structure
        str_new = open('MoREST.str_new','w')
        write_xyz_file(str_new, self.structure)
        if self.structure['current_step'] % self.sampling_parameters['sampling_traj_interval'] == 0:
            write_xyz_file(self.__log_traj, self.structure)
        
        return self.structure
    
        
    def get_current_structure(self):
        if self.sampling_parameters['sampling_restart']:
            structure = read_xyz_file('MoREST.str_new')
        else:
            structure = read_xyz_file('MoREST.str')
            structure['current_step'] = 0
            structure['total_energy'], structure['forces'] = ml_potential().get_potential_FD_forces(structure['coordinates'])
            structure['accelerations'] = np.array([structure['forces'][i_atom] / structure['masses'][i_atom] \
                                              for i_atom in range(len(structure['forces']))])            
        return structure
    