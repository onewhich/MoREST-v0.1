#from time import time
import os
import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_traj
from initialize_calculator import initialize_calculator
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
from kinetic_energy_assignment import clean_translation, clean_rotation, clean_translation_vm
from coordinates_operation import reset_mass_center
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
                self.ml_calculator.set_current_step(self.current_step)
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
        reset_mass_center(incident_molecule)
        clean_translation(target_molecule)
        reset_mass_center(target_molecule)

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
            if 'scattering_fix_target_atoms' in self.scattering_parameters:
                self.index_fix_target_atoms = self.scattering_parameters['scattering_fix_target_atoms']
            else:
                self.index_fix_target_atoms = np.arange(0, target_molecule.get_global_number_of_atoms(), dtype=int)
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
            next_velocities[self.index_fix_target_atoms] = clean_translation_vm(next_velocities[self.index_fix_target_atoms], masses)
            self.current_system.set_velocities(next_velocities)
        
        self.current_system.info['step'] = self.current_step
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.set_current_step(self.current_step)
        except:
            pass
        
        self.current_step += 1
    
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
