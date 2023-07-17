#from time import time
import numpy as np
from structure_io import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from initialization import initialize_calculator
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units

class initialize_scattering(initialize_calculator):
    '''
    The mass center of target molecule locates at original point [0,0,0].
    The mass center of incident molecule locates on the spherical surface with a specified radius and centered at original point.
    The incident momenta directs from the mass center of incident molecule to the point on a spherical surface closely covering the target molecule and centered at original point.
    The target molecule is in the front of the incident molecule in the combined scattering system.
    '''
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, i_traj=0, log_morest=None):
        super(initialize_scattering, self).__init__(morest_parameters, calculator, log_morest)
        self.scattering_parameters = scattering_parameters
        traj_filename = 'MoREST_traj_'+str(i_traj)+'.xyz'
        log_filename = 'MoREST_log_'+str(i_traj)+'.log'
            
        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.scattering_parameters['scattering_T_target'] # Ek = 1/2 m v^2 = 3/2 kB T for each particle

        if self.scattering_parameters['scattering_initialization']:
            self.generate_scattering_system()
            self.current_step = 0
            self.current_system = self.get_current_structure()
            #self.current_traj = []
            #self.current_traj.append(self.current_system)
            write_xyz_traj(traj_filename, self.current_system)
            self.MD_log = open(log_filename, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            try:
                self.current_traj = read_xyz_traj(traj_filename)
                self.current_step = len(self.current_traj) - 1
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
                self.MD_log = open(log_filename, 'a', buffering=1)
            except:
                self.generate_scattering_system()
                self.current_step = 0
                self.current_system = self.get_current_structure()
                #self.current_traj = []
                #self.current_traj.append(self.current_system)
                write_xyz_traj(traj_filename, self.current_system)
                self.MD_log = open(log_filename, 'w', buffering=1)
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
                write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        
    def generate_scattering_system(self):
        if self.scattering_parameters['scattering_pre_thermolized']:
            pass
        else:
            target_molecule = read_xyz_file(self.scattering_parameters['scattering_target_molecule'])
            MaxwellBoltzmannDistribution(target_molecule, temperature_K = self.scattering_parameters['scattering_T_target'])
            Stationary(target_molecule)
            reset_mass_center(target_molecule)

            incident_molecule = read_xyz_file(self.scattering_parameters['scattering_incident_molecule'])
            MaxwellBoltzmannDistribution(incident_molecule, temperature_K = self.scattering_parameters['scattering_T_incident'])
            Stationary(incident_molecule)
            reset_mass_center(incident_molecule)
        # set collision momentum
        #scalar_translational_momentum = np.linalg.norm(get_translational_momentum(incident_molecule))
        target_point = np.random.uniform(-1,1,3)
        target_point = self.scattering_parameters['scattering_R_target'] * target_point / np.linalg.norm(target_point)
        incident_point = np.random.uniform(-1,1,3)
        incident_point = self.scattering_parameters['scattering_R_incident'] * incident_point / np.linalg.norm(incident_point)
        # normalized collision_vector
        collision_vector = (target_point - incident_point) / np.linalg.norm(target_point - incident_point)
        # if Scattering_E_collision is given, Scattering_V_collision will be ignored.
        if 'scattering_E_collision' in self.scattering_parameters:
            collision_velocity = collision_vector * np.sqrt( 2*self.scattering_parameters['scattering_E_collision'] / np.sum(incident_molecule.get_masses()) )
        else:
            collision_velocity = collision_vector * self.scattering_parameters['scattering_V_collision']
        incident_molecule.set_velocities(incident_molecule.get_velocities() + collision_velocity)
        # move the mass center of incident molecule to the incident_point
        incident_molecule.set_positions(incident_molecule.get_positions() + incident_point)

        # combine target molecule and incident molecule
        scattering_system = target_molecule + incident_molecule
        write_xyz_file('MoREST.str', scattering_system)
            
    def get_current_structure(self):
        if self.scattering_parameters['scattering_initialization']:
            system = read_xyz_file('MoREST.str')
        else:
            try:
                system = self.current_traj[-1]
            except:
                system = read_xyz_file('MoREST.str')
            
        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        
        return system
    

class scattering_velocity_Verlet(initialize_scattering):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, i_traj=0, log_morest=None):
        super(scattering_velocity_Verlet, self).__init__(morest_parameters, scattering_parameters, calculator, i_traj, log_morest)
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.scattering_parameters['scattering_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces
        
        ### x(t), v(t) = p(t) / m
        current_coordinates = self.current_system.get_positions()
        current_momenta = self.current_system.get_momenta()
        
        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        next_coordinates = current_coordinates + (current_momenta * time_step + 0.5 * self.current_forces * time_step**2) / self.masses
        self.current_system.set_positions(next_coordinates)
        
        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * self.current_forces * time_step
        
        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        self.current_system.set_momenta(next_momenta)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        #self.current_traj.append(self.current_system)
        write_xyz_traj(traj_filename, self.current_system)
        kinetic_energy = self.current_system.get_kinetic_energy()
        write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system

class scattering_Runge_Kutta_4th(initialize_scattering):
    '''
    This class implements Runge-Kutta 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, i_traj=0, log_file=None):
        super(scattering_Runge_Kutta_4th, self).__init__(morest_parameters, scattering_parameters, calculator, i_traj, log_file)
        
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
        
        # x_1 = x_n, v_1 = v_n, a_1 = a_n
        x_1 = self.current_system.get_positions()
        v_1 = self.current_system.get_velocities()
        a_1 = self.current_forces/self.masses

        # x_2 = x_n + h/2 * v_1, v_2 = v_n + h/2 * a_1, a_2 = f(x_2)
        x_2 = x_1 + time_step/2 * v_1
        v_2 = v_1 + time_step/2 * a_1
        self.current_system.set_positions(x_2)
        Ep_2, F_2 = self.many_body_potential.get_potential_forces(self.current_system)
        a_2 = F_2/self.masses

        # x_3 = x_n + h/2 * v_2, v_3 = v_n + h/2 * a_2, a_3 = f(x_3)
        x_3 = x_1 + time_step/2 * v_2
        v_3 = v_1 + time_step/2 * a_2
        self.current_system.set_positions(x_3)
        Ep_3, F_3 = self.many_body_potential.get_potential_forces(self.current_system)
        a_3 = F_3/self.masses

        # x_4 = x_n + h * v_3, v_4 = v_n + h * a_3, a_4 = f(x_4)
        x_4 = x_1 + time_step * v_3
        v_4 = v_1 + time_step * a_3
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
        
        #self.current_traj.append(self.current_system)
        write_xyz_traj(traj_filename, self.current_system)
        kinetic_energy = self.current_system.get_kinetic_energy()
        write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system
 
# Discarded: the following method is not energy preseved.
class scattering_Runge_Kutta_4th_a(initialize_scattering):
    '''
    This class implements Runge-Kutta 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, scattering_parameters, calculator=None, log_file=None):
        super(scattering_Runge_Kutta_4th_a, self).__init__(morest_parameters, scattering_parameters, calculator, log_file)
        
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
        
        #self.current_traj.append(self.current_system)
        write_xyz_traj(traj_filename, self.current_system)
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
