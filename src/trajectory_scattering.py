#from time import time
import os
import numpy as np
from structure_io import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from initialize_calculator import initialize_calculator
from numerical_integraion import MD_integration
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
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
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.scattering_log = open('MoREST_scattering.log', 'a', buffering=1)

            
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
        scalar_translational_velocity = np.linalg.norm(get_translational_velocity(incident_molecule) - get_translational_velocity(target_molecule))
        collision_energy = 0.5 * np.sum(incident_molecule.get_masses()) * scalar_translational_velocity**2
        
        # initialize incident and target molecules
        incident_molecule.set_velocities(self.clean_translation(incident_molecule.get_velocities()))
        reset_mass_center(incident_molecule)
        target_molecule.set_velocities(self.clean_translation(target_molecule.get_velocities()))
        reset_mass_center(target_molecule)
        if self.scattering_parameters['scattering_clean_rotation']:
            incident_molecule.set_velocities(self.clean_rotation(incident_molecule.get_velocities(), incident_molecule.get_positions(), incident_molecule.get_masses()))
            target_molecule.set_velocities(self.clean_rotation(target_molecule.get_velocities(), target_molecule.get_positions(), target_molecule.get_masses()))

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
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
            
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
    
    @staticmethod
    def rescale_T_kinetic(system, Tf):
        n_atom = system.get_global_number_of_atoms()
        Ek_i = system.get_kinetic_energy()
        Ti = 2/3 * Ek_i/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        velocities = system.get_velocities()
        factor = np.sqrt(Tf / Ti)
        system.set_velocities(factor * velocities)
        return system
    
    @staticmethod
    def clean_translation(velocities):
        total_velocity = np.sum(velocities, axis=0)/len(velocities)
        new_velocities = velocities - total_velocity
        return new_velocities
        
    @staticmethod
    def clean_rotation(velocities, coordinates, masses):
        '''
        L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
        L : angular momentum
        omega: angular velocity
        I : moment of inertia
        '''
        n_atom = len(velocities)
        if n_atom == 1:
            return velocities
        masses = masses[:,np.newaxis]
        v_vector = velocities
        #center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
        #center_of_mass = np.sum(masses[:,np.newaxis]*coordinates, axis=0)/np.sum(masses)
        center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
        r_vector = coordinates - center_of_mass
        # r_cross_v : angular velocities
        # omega = (r x v) / |r|^2
        r_cross_v = np.cross(r_vector, v_vector)
        r_2 = np.linalg.norm(r_vector, axis=1)**2
        omega = np.array([r_cross_v[i]/r_2[i] for i in range(n_atom)])
        # Rv = omega/n_atom : system total angular velocity
        rotat_vector = np.sum(omega, axis=0)/n_atom
        v_tang = np.cross(rotat_vector, r_vector)
        new_velocities = v_vector - v_tang
        return new_velocities
    

class scattering_velocity_Verlet(initialize_scattering):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, log_morest=None):
        super(scattering_velocity_Verlet, self).__init__(morest_parameters, scattering_parameters, calculator, log_morest)
        self.integration = MD_integration(self.many_body_potential)
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.scattering_parameters['scattering_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces

        next_potential_energy, next_forces  = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        if self.scattering_parameters['scattering_fix_target']:
            next_momenta = self.current_system.get_momenta()
            next_momenta[0:self.n_atom_target] = self.clean_translation(next_momenta[0:self.n_atom_target])
            self.current_system.set_momenta(next_momenta)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.scattering_parameters['scattering_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_filename, self.current_system)
            kinetic_energy = self.current_system.get_kinetic_energy()
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system

class scattering_Runge_Kutta_4th(initialize_scattering):
    '''
    This class implements Runge-Kutta 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, log_morest=None):
        super(scattering_Runge_Kutta_4th, self).__init__(morest_parameters, scattering_parameters, calculator, log_morest)
        self.integration = MD_integration(self.many_body_potential)
        
    def generate_new_step(self, bias_forces=None):
        '''
        This version comes from classic Runge-Kutta methods:
        Runge–Kutta methods. (2022, September 6). In Wikipedia. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
         https://www.haroldserrano.com/blog/visualizing-the-runge-kutta-method
        '''
        time_step = self.scattering_parameters['scattering_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces
        
        next_potential_energy, next_forces  = self.integration.Runge_Kutta_4th(time_step, self.current_system, self.current_forces, self.masses)
        if self.scattering_parameters['scattering_fix_target']:
            next_momenta = self.current_system.get_momenta()
            next_momenta[0:self.n_atom_target] = self.clean_translation(next_momenta[0:self.n_atom_target])
            self.current_system.set_momenta(next_momenta)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.scattering_parameters['scattering_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_filename, self.current_system)
            kinetic_energy = self.current_system.get_kinetic_energy()
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system
 
# Discarded: the following method is not energy preseved.
class scattering_Runge_Kutta_4th_a(initialize_scattering):
    '''
    This class implements Runge-Kutta 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, log_morest=None):
        super(scattering_Runge_Kutta_4th_a, self).__init__(morest_parameters, scattering_parameters, calculator, log_morest)
        
    def generate_new_step(self, bias_forces=None):
        '''
        This Runge-Kutta-Nyström methods version comes from https://willbeason.com/2021/06/24/introduction-to-runge-kutta-nystrom-methods/
        '''
        time_step = self.scattering_parameters['scattering_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces
        
        # x_1 = x_n, v_1 = v_n, a_1 = a_n
        x_1 = self.current_system.get_positions()
        v_1 = self.current_system.get_velocities()
        a_1 = self.current_forces/self.masses

        # x_2 = x_n + h/2 * v_1, v_2 = v_n + h/2 * a_1, a_2 = f(x_2)
        v_2 = v_1 + time_step/2 * a_1
        x_2 = x_1 + time_step/2 * (v_1 + v_2)/2
        self.current_system.set_positions(x_2)
        Ep_2, F_2 = self.many_body_potential.get_potential_forces(self.current_system)
        a_2 = F_2/self.masses

        # x_3 = x_n + h/2 * v_2, v_3 = v_n + h/2 * a_2, a_3 = f(x_3)
        v_3 = v_1 + time_step/2 * a_2
        x_3 = x_1 + time_step/2 * (v_1 + v_3)/2
        self.current_system.set_positions(x_3)
        Ep_3, F_3 = self.many_body_potential.get_potential_forces(self.current_system)
        a_3 = F_3/self.masses

        # x_4 = x_n + h * v_3, v_4 = v_n + h * a_3, a_4 = f(x_4)
        v_4 = v_1 + time_step * a_3
        x_4 = x_1 + time_step * (v_1 + v_4)/2
        self.current_system.set_positions(x_4)
        Ep_4, F_4 = self.many_body_potential.get_potential_forces(self.current_system)
        a_4 = F_4/self.masses

        # x_n+1 = x_n + h/6 * (v_1 + 2*v_2 + 2*v_3 + v_4), v_n+1 = v_n + h/6 * (a_1 + 2*a_2 + 2*a_3 + a_4)
        next_coordinates = x_1 + time_step/6 * (v_1 + 2*v_2 + 2*v_3 + v_4)
        next_velocities = v_1 + time_step/6 * (a_1 + 2*a_2 + 2*a_3 + a_4)

        self.current_system.set_positions(next_coordinates)
        self.current_system.set_velocities(next_velocities)

        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.scattering_parameters['scattering_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_filename, self.current_system)
            kinetic_energy = self.current_system.get_kinetic_energy()
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system

def reset_geometric_center(system):
    '''
    set the geometric center to [0,0,0]
    '''
    coordinates = system.get_positions()
    n_atom = system.get_global_number_of_atoms()
    geometric_center = np.sum(coordinates, axis=0)/n_atom
    system.set_positions(coordinates - geometric_center)

def reset_mass_center(system):
    '''
    set the mass center to [0,0,0]
    '''
    coordinates = system.get_positions()
    masses = system.get_masses()[:,np.newaxis]
    mass_center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    system.set_positions(coordinates - mass_center)

def get_translational_momentum(system):
    n_atom = system.get_global_number_of_atoms()
    return np.sum(system.get_momenta(), axis=0)/n_atom

def get_translational_velocity(system):
    n_atom = system.get_global_number_of_atoms()
    return np.sum(system.get_velocities(),axis=0)/n_atom

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
    elif center.upper() == 'geometry'.upper():
        center = np.sum(coordinates, axis=0)/len(coordinates)
    elif center.upper() == 'mass'.upper():
        masses = system.get_masses()[:,np.newaxis]
        center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    coordinates = coordinates - center

    r_cross_r_new = np.sin(theta) * unit_normal_vector
    r = np.linalg.norm(coordinates, axis=1)[:,np.newaxis]
    r2 = r**2
    r_cross_r_new = r_cross_r_new * r2

    system_new = system.copy()
    coordinates_new = np.cross(r_cross_r_new, coordinates) / r2 + np.cos(theta) * coordinates
    system_new.set_positions(coordinates_new)

    return system_new

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
