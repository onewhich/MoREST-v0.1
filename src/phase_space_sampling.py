import numpy as np
#import sys
#sys.path.append('..')
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_potential
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

class velocity_Verlet:
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str records the initial xyz structure of the system
    MoREST.str_new records the current xyz structure of the system
    '''
    
    def __init__(self, sampling_parameters, md_parameters):
        #self.md_parameters = np.load('MoREST_md_parameters.npy',allow_pickle=True).item()
        self.sampling_parameters = sampling_parameters
        self.md_parameters = md_parameters
        
        if self.sampling_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            trained_ml_potential = self.sampling_parameters['ml_potential_model']
            self.many_body_potential = ml_potential(trained_ml_potential)
        else:
            raise Exception('Which many body potential will you use?')
        
        if self.sampling_parameters['sampling_initialization']:
            self.current_step = 0
            self.current_step, self.current_system = self.get_current_structure()
            if self.md_parameters['md_temperature'] > 1e-6:
                MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.md_parameters['md_temperature'])
            self.current_traj = []
            self.current_traj.append(self.current_system)
            write_xyz_traj('MoREST_traj.xyz', self.current_system)
            
            self.MD_log = open('MoREST_MD.log', 'w')
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_velocities(), self.masses)
            
        else:
            self.current_traj = read_xyz_traj('MoREST_traj.xyz')
            self.current_step = len(self.current_traj) - 1
            self.current_step, self.current_system = self.get_current_structure()
            
            self.MD_log = open('MoREST_MD.log', 'a')
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.md_parameters['md_time_step']
        
        next_system = deepcopy(self.current_system)
        
        current_coordinates = self.current_system.get_positions()
        current_velocities = self.current_system.get_velocities()
        next_coordinates = current_coordinates + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_system.set_positions(next_coordinates)
        
        next_potential_energy, next_forces = self.many_body_potential.get_potential_FD_forces(next_system, \
                                                                 self.sampling_parameters['fd_displacement'])
        if type(bias_forces) != type(None):
            next_forces = next_forces + bias_forces        
        
        next_accelerations = np.array([next_forces[i_atom] / self.masses[i_atom] for i_atom in range(len(self.masses))])
        next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        
        if self.sampling_parameters['sampling_clean_translation']:
            next_velocities = self.clean_translation(next_velocities)
        if self.sampling_parameters['sampling_clean_rotation']:
            next_velocities = self.clean_rotation(next_velocities, next_coordinates, self.masses)
        
        next_system.set_velocities(next_velocities)
        
        self.current_step = self.current_step + 1
        self.current_system = next_system
        
        write_xyz_file('MoREST.str_new', next_system)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            self.current_traj.append(next_system)
            write_xyz_traj('MoREST_traj.xyz', next_system)
            self.write_MD_log(self.MD_log, self.current_step, next_potential_energy, next_velocities, self.masses)
        
        return next_system
    
        
    def get_current_structure(self):
        if self.sampling_parameters['sampling_initialization']:
            system = read_xyz_file('MoREST.str')
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_FD_forces(system, \
                                                                 self.sampling_parameters['fd_displacement'])
            self.masses = system.get_masses()
            self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(len(self.masses))])
        else:
            system = read_xyz_file('MoREST.str_new')
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_FD_forces(system, \
                                                                 self.sampling_parameters['fd_displacement'])
            self.masses = system.get_masses()
            self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(len(self.masses))])
        return self.current_step, system
    
    @staticmethod
    def clean_translation(velocities):
        total_velocity = np.sum(velocities, axis=0)/len(velocities)
        velocities = velocities - total_velocity
        return velocities
    
    @staticmethod
    def clean_rotation(velocities, coordinates, masses):
        v_vector = velocities
        center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
        r_vector = coordinates - center_of_mass
        
        r_cross_v = np.cross(r_vector, v_vector)
        r_2 = np.linalg.norm(r_vector, axis=1)**2
        omega = np.array([r_cross_v[i]/r_2[i] for i in range(4)])
        rotat_vector = np.sum(omega, axis=0)/len(masses)
        v_tang = np.cross(rotat_vector, r_vector)
        velocities = v_vector - v_tang
        
        return velocities
        
    @staticmethod
    def write_MD_log(MD_log, step, Ep, velocities, masses):
        Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(len(masses))])
        T = 2/3 * Ek/units.kB    # Ek = 1/2 m v^2 = 3/2 kB T
        Et = Ek + Ep
        MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
