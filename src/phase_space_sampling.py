import numpy as np
#import sys
#sys.path.append('..')
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_potential
from copy import deepcopy

class velocity_Verlet:
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling.
    MoREST.xyz_traj records the trajectory in an extended xyz format
    MoREST.str records the initial xyz structure of the system
    MoREST.str_new records the current xyz structure of the system
    '''
    
    def __init__(self, sampling_parameters, md_parameters):
        #self.md_parameters = np.load('MoREST_md_parameters.npy',allow_pickle=True).item()
        self.sampling_parameters = sampling_parameters
        self.md_parameters = md_parameters
        
        if self.sampling_parameters['sampling_restart']:
            self.current_traj = read_xyz_traj('MoREST.xyz_traj')
            self.current_step = len(self.current_traj) - 1
        else:
            self.current_traj = []
            self.current_traj.append(current_system)
            self.current_step = 0
            
        self.current_system = self.get_current_system()
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.md_parameters['md_time_step']
        
        next_system = deepcopy(self.current_system)
        
        current_coordinates = self.current_system.get_positions()
        current_velocities = self.current_system.get_velocities()
        next_coordinates = current_coordinates + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_system.set_positions(next_coordinates)
        
        next_potential_energy, next_forces = ml_potential().get_potential_FD_forces(next_system)
        if bias_forces != None:
            next_forces = forces + bias_forces        
        
        next_accelerations = np.array([next_forces[i_atom] / self.masses[i_atom] for i_atom in range(len(masses))])
        next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        next_system.set_velocities(next_velocities)
        
        self.current_step = self.current_step + 1
        
        write_xyz_file('MoREST.str_new', self.next_system)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            self.current_traj.append(self.next_system)
            write_xyz_traj('MoREST.xyz_traj', self.current_traj)
        
        return self.next_system
    
        
    def get_current_structure(self):
        if self.sampling_parameters['sampling_restart']:
            system = read_xyz_file('MoREST.str_new')
            self.current_potential_energy, self.current_forces = ml_potential().get_potential_FD_forces(system)
            masses = system.get_masses()
            self.current_accelerations = np.array([self.current_forces[i_atom] / masses[i_atom] for i_atom in range(len(masses))])   
        else:
            system = read_xyz_file('MoREST.str')
            self.current_potential_energy, self.current_forces = ml_potential().get_potential_FD_forces(system)
            self.masses = system.get_masses()
            self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(len(masses))])
        return self.current_step, system
    