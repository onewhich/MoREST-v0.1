#from time import time
import os
import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_traj
from initialize_calculator import initialize_calculator
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
from kinetic_energy_assignment import clean_translation, clean_rotation, clean_translation_vm
from numerical_integration import MD_integration
from ase import units

class initialize_scattering(initialize_calculator):
    '''
    The mass center of target molecule locates at original point [0,0,0].
    The mass center of incident molecule locates on the spherical surface with a specified radius and centered at original point.
    The incident momenta directs from the mass center of incident molecule to the point on a spherical surface closely covering the target molecule and centered at original point.
    The target molecule is in the front of the incident molecule in the combined scattering system.
    '''
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, log_morest=None):
        super(initialize_scattering, self).__init__(morest_parameters, calculator, log_morest)
        self.scattering_parameters = scattering_parameters
        
        if self.scattering_parameters['scattering_initialization']:
            self.scattering_log = open('MoREST_scattering.log', 'w', buffering=1)
            self.scattering_log.write('# traj number, impact parameter (A), collision energy (eV)\n')

            i_traj = 0
            self.traj_filename = 'MoREST_scattering_traj_'+str(i_traj)+'.xyz'
            log_filename = 'MoREST_scattering_traj_'+str(i_traj)+'.log'

            self.generate_scattering_system(i_traj)
            self.current_traj = []
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure()
            self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_filename, self.current_system)
            self.MD_log = open(log_filename, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.scattering_log = open('MoREST_scattering.log', 'a', buffering=1)

        self.integration = MD_integration(self.many_body_potential)
        ### kinetic energy at simulation temperature
        #Nf = 3 * self.n_atom
        #self.K_simulation = Nf/2 * units.kB * self.scattering_parameters['scattering_T_target'] # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    
    
    def generate_scattering_system(self, i_traj):
        if self.scattering_parameters['scattering_pre_thermalized']:
            incident_molecule = read_xyz_file(self.scattering_parameters['scattering_incident_molecule'])
            target_molecule = read_xyz_file(self.scattering_parameters['scattering_target_molecule'])
        else:
            incident_molecule = read_xyz_file(self.scattering_parameters['scattering_incident_molecule'])
            MaxwellBoltzmannDistribution(incident_molecule, temperature_K = self.scattering_parameters['scattering_T_incident'], \
                                         force_temp = self.scattering_parameters['scattering_T_kinetic'])
            
            target_molecule = read_xyz_file(self.scattering_parameters['scattering_target_molecule'])
            MaxwellBoltzmannDistribution(target_molecule, temperature_K = self.scattering_parameters['scattering_T_target'], \
                                         force_temp = self.scattering_parameters['scattering_T_kinetic'])


        # get collision velocity
        scalar_translational_velocity = np.linalg.norm(self.get_translational_velocity(incident_molecule) - self.get_translational_velocity(target_molecule))
        collision_energy = 0.5 * np.sum(incident_molecule.get_masses()) * scalar_translational_velocity**2
        
        # initialize incident and target molecules
        if self.scattering_parameters['scattering_clean_rotation']:
            clean_rotation(incident_molecule)
            clean_rotation(target_molecule)
        clean_translation(incident_molecule)
        self.reset_mass_center(incident_molecule)
        clean_translation(target_molecule)
        self.reset_mass_center(target_molecule)

        # uniform sampling on a sphere for inciden point and on a disc for target point.
        if not self.scattering_parameters['scattering_fix_incident']:
            # sampling spherical coordinate system (r,theta,phi), angle in radians.
            # x = r * sin(theta) * cos(phi)
            # y = r * sin(theta) * sin(phi)
            # z = r * cos(theta)
            s_theta = np.random.uniform(0,np.pi)
            s_phi = np.random.uniform(0,2*np.pi)
            s_r = self.scattering_parameters['scattering_R_incident']
            incident_point = np.array([s_r*np.sin(s_theta)*np.cos(s_phi), s_r*np.sin(s_theta)*np.sin(s_phi), s_r*np.cos(s_theta)])
        else:
            incident_point = np.array([0.0, 0.0, self.scattering_parameters['scattering_R_incident']])
        if self.scattering_parameters['scattering_fix_target']:
            self.n_atom_target = target_molecule.get_global_number_of_atoms()
        # the plane including the disc is perpendicular to the vector from incident point to the coordinate origin.
        # the plane is formed with the normal vector (a,b,c) and the point (x1,y1,z1) on the plane.
        # the plane formular is a(x-x1) + b(y-y1) + c(z-z1) = 0
        # first generate a uniform sampling on the plane, then screen out the samples on the disc.
        [nv_a,nv_b,nv_c] = incident_point
        d_r = self.scattering_parameters['scattering_R_target']
        if d_r < 0.1:
            target_point = np.array([0.0,0.0,0.0])
        elif nv_a > 1e-4:
            while True:
                [p_y,p_z] = np.random.uniform(-d_r,d_r,2)
                p_x = -(nv_b*p_y + nv_c*p_z) / nv_a
                target_point = np.array([p_x,p_y,p_z])
                if np.linalg.norm(target_point) < d_r:
                    break
        elif nv_b > 1e-4:
            while True:
                [p_x,p_z] = np.random.uniform(-d_r,d_r,2)
                p_y = -(nv_a*p_x + nv_c*p_z) / nv_b
                target_point = np.array([p_x,p_y,p_z])
                if np.linalg.norm(target_point) < d_r:
                    break
        else:
            while True:
                [p_x,p_y] = np.random.uniform(-d_r,d_r,2)
                p_z = -(nv_a*p_x + nv_b*p_y) / nv_c
                target_point = np.array([p_x,p_y,p_z])
                if np.linalg.norm(target_point) < d_r:
                    break
        
        # normalized collision_vector
        collision_vector = (target_point - incident_point) / np.linalg.norm(target_point - incident_point)
        
        # calculate the impact parameter (ip): the distance of point (x0,y0,z0) to the collision line.
        # the collision line is formed with the vector (m,n,p) and the online point (x1,y1,z1).
        # the collision line formular is (x-x1)/m = (y-y1)/n = (z-z1)/p = t,
        # where t = [m*(x0-x1)+n*(y0-y1)+p*(z0-z1)]/(m*m+n*n+p*p).
        # and point (xf,yf,zf) is the perpendicular foot of point (x0,y0,z0) on the collision line,
        # where xf = m*t+x1, yf = n*t+y1, zf = p*t+z1.
        # the distance is the norm of (x0-xf,y0-yf,z0-zf).
        # point (x0,y0,z0) is the coordinate origin point (0,0,0) here.
        [ip_m,ip_n,ip_p] = collision_vector
        [ip_x1,ip_y1,ip_z1] = incident_point
        ip_t = (-ip_m*ip_x1-ip_n*ip_y1-ip_p*ip_z1)/(ip_m*ip_m+ip_n*ip_n+ip_p*ip_p)
        impact_parameter = np.linalg.norm([ip_m*ip_t+ip_x1, ip_n*ip_t+ip_y1, ip_p*ip_t+ip_z1])
        
        # if Scattering_E_collision is given, Scattering_V_collision will be ignored.
        # if scattering_Maxwell_Boltzmann_collision is True, scattering_E_collision and scattering_V_collision will be ignored
        if self.scattering_parameters['scattering_Maxwell_Boltzmann_collision'] == True:
            collision_velocity = collision_vector * scalar_translational_velocity
        elif 'scattering_E_collision' in self.scattering_parameters:
            collision_velocity = collision_vector * np.sqrt( 2*self.scattering_parameters['scattering_E_collision'] / np.sum(incident_molecule.get_masses()) )
            collision_energy = self.scattering_parameters['scattering_E_collision']
        else:
            collision_velocity = collision_vector * self.scattering_parameters['scattering_V_collision']
            collision_energy = 0.5 * np.sum(incident_molecule.get_masses()) * self.scattering_parameters['scattering_V_collision']**2
        incident_molecule.set_velocities(incident_molecule.get_velocities() + collision_velocity)
        
        # move the mass center of incident molecule to the incident_point
        incident_molecule.set_positions(incident_molecule.get_positions() + incident_point)

        # combine target molecule and incident molecule
        self.scattering_system = target_molecule + incident_molecule
        #write_xyz_file('MoREST_scattering.xyz', self.scattering_system)

        self.scattering_log.write(str(i_traj)+'    '+str(impact_parameter)+'    '+str(collision_energy)+'\n')

    def generate_new_traj(self, i_traj):
        self.traj_filename = 'MoREST_scattering_traj_'+str(i_traj)+'.xyz'
        log_filename = 'MoREST_scattering_traj_'+str(i_traj)+'.log'
        
        if os.path.isfile(self.traj_filename):
            self.current_traj = read_xyz_traj(self.traj_filename)
            self.current_step = len(self.current_traj) - 1
            self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.xyz_new instead of MoREST_scattering_traj.xyz
            self.MD_log = open(log_filename, 'a', buffering=1)
        else:
            self.generate_scattering_system(i_traj)
            self.current_traj = []
            self.current_step = 0
            self.current_system = self.get_current_structure()
            self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_filename, self.current_system)
            self.MD_log = open(log_filename, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
            
    def get_current_structure(self):
        if self.scattering_parameters['scattering_initialization']:
            system = self.scattering_system
        else:
            if len(self.current_traj) == 0:
                system = self.scattering_system
            else:
                system = self.current_traj[-1]

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)

        return system

    def update_pre_step(self, time_step=None, bias_forces=None):
        if type(time_step) == type(None):
            time_step = self.scattering_parameters['scattering_time_step']

        ### F(t) + bias
        if type(bias_forces) != type(None):
            self.current_forces = self.current_forces + bias_forces

        return time_step
        
    def update_step(self, next_potential_energy, next_forces):

        if self.scattering_parameters['scattering_fix_target']:
            next_velocities = self.current_system.get_velocities()
            masses = self.current_system.get_masses()
            next_velocities[0:self.n_atom_target] = clean_translation_vm(next_velocities[0:self.n_atom_target], masses)
            self.current_system.set_velocities(next_velocities)
        
        self.current_system.info['step'] = self.current_step
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        self.current_step += 1
    
    @staticmethod
    def reset_geometric_center(system):
        '''
        set the geometric center to [0,0,0]
        '''
        coordinates = system.get_positions()
        n_atom = system.get_global_number_of_atoms()
        geometric_center = np.sum(coordinates, axis=0)/n_atom
        system.set_positions(coordinates - geometric_center)

    @staticmethod
    def reset_mass_center(system):
        '''
        set the mass center to [0,0,0]
        '''
        coordinates = system.get_positions()
        masses = system.get_masses()[:,np.newaxis]
        mass_center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
        system.set_positions(coordinates - mass_center)

    @staticmethod
    def get_translational_momentum(system):
        n_atom = system.get_global_number_of_atoms()
        return np.sum(system.get_momenta(), axis=0)/n_atom

    @staticmethod
    def get_translational_velocity(system):
        n_atom = system.get_global_number_of_atoms()
        return np.sum(system.get_velocities(),axis=0)/n_atom

    @staticmethod
    def rotate_system_at_center(system, theta, unit_normal_vector, center=[0,0,0]):
        '''
        theta: angles, in degree
        center: 1-D list of 3 numbers (define the position of the center), or 'geometry' (calculate geometric center), or 'mass' (calculate mass' center)
        r x r' = n,
        when r and n are orthogonal, r' = (n x r) / (|r|*|n|) * r +  cos(theta) * r
        '''
        # (r x r_new) / (|r| * |r_new|) = sin(theta) * unit_normal_vector, where |r| == |r_new|
        # r x r_new = sin(theta) * unit_normal_vector * |r|^2
        theta = np.deg2rad(theta)
        unit_normal_vector = np.array(unit_normal_vector)
        unit_normal_vector = unit_normal_vector / np.linalg.norm(unit_normal_vector)
        coordinates = system.get_positions()
        if type(center) == list:
            center = np.array(center)
        elif center.lower() == 'geometry':
            center = np.sum(coordinates, axis=0)/len(coordinates)
        elif center.lower() == 'mass':
            #masses = system.get_masses()[:,np.newaxis]
            #center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
            center = system.get_center_of_mass()
        coordinates = coordinates - center

        r_cross_r_new = np.sin(theta) * unit_normal_vector
        r = np.linalg.norm(coordinates, axis=1)[:,np.newaxis]
        r2 = r**2
        r_cross_r_new = r_cross_r_new * r2

        system_new = system.copy()
        coordinates_new = np.cross(r_cross_r_new, coordinates) / r2 + np.cos(theta) * coordinates
        system_new.set_positions(coordinates_new)

        return system_new

    def rotate_at_center(coordinates, theta, unit_normal_vector, center=np.array([0, 0, 0], dtype=float)):
        """
        Rotates a set of coordinates around a specified center point
        by an angle theta about an arbitrary axis defined by a unit normal vector.

        Args:
            coordinates (np.ndarray): An N x 3 numpy array of points to rotate.
            theta (float): The angle of rotation in radians.
            unit_normal_vector (np.ndarray): A 3-element numpy array representing the
                                            unit vector of the rotation axis.
            center (np.ndarray, optional): A 3-element numpy array representing the
                                        center of rotation. Defaults to [0,0,0]
                                        if None is provided.

        Returns:
            np.ndarray: The N x 3 numpy array of rotated coordinates.
        """

        center = np.array(center, dtype=float)

        unit_normal_vector = np.array(unit_normal_vector, dtype=float)
        
        # Ensure the normal vector is a unit vector
        norm_n = np.linalg.norm(unit_normal_vector)
        if norm_n == 0:
            raise ValueError("The unit_normal_vector cannot be a zero vector.")
        unit_normal_vector = unit_normal_vector / norm_n

        # Convert coordinates to numpy array and shift to origin for rotation
        coordinates = np.array(coordinates, dtype=float)
        shifted_coordinates = coordinates - center

        # Apply Rodrigues' Rotation Formula
        # v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))

        # Component 1: v * cos(theta)
        term1 = shifted_coordinates * np.cos(theta)

        # Component 2: (k x v) * sin(theta)
        # np.cross handles broadcasting when one argument has more dimensions
        term2 = np.cross(unit_normal_vector, shifted_coordinates) * np.sin(theta)

        # Component 3: k * (k . v) * (1 - cos(theta))
        # np.dot(shifted_coordinates, unit_normal_vector) computes the dot product
        # for each row of shifted_coordinates with unit_normal_vector
        term3_scalar = np.dot(shifted_coordinates, unit_normal_vector) * (1 - np.cos(theta))
        term3 = unit_normal_vector * term3_scalar[:, np.newaxis] # Reshape term3_scalar for broadcasting

        # Sum the terms
        rotated_coordinates = term1 + term2 + term3

        # Shift coordinates back from origin
        rotated_coordinates_shifted_back = rotated_coordinates + center

        return rotated_coordinates_shifted_back

    @staticmethod
    def write_MD_log(MD_log, step, Ep, Ek, masses):
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
        except:
            pass
        n_atom = len(masses)
        #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
        #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
        T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        Et = Ek + Ep
        MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
