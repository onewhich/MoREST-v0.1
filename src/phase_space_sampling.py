import numpy as np
#import sys
#sys.path.append('..')
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_potential, on_the_fly
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units

class initialize_sampling:
    def __init__(self, sampling_parameters, md_parameters, calculator=None):
        #self.md_parameters = np.load('MoREST_md_parameters.npy',allow_pickle=True).item()
        self.sampling_parameters = sampling_parameters
        self.md_parameters = md_parameters
        
        if self.sampling_parameters['many_body_potential'].upper() in ['on_the_fly'.upper()]:
            if type(calculator) == type(None):
                raise Exception('Please specify the electronic structure method.')
            self.many_body_potential = on_the_fly(calculator)
        elif self.sampling_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
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
                    
        else:
            self.current_traj = read_xyz_traj('MoREST_traj.xyz')
            self.current_step = (len(self.current_traj) - 1) * self.sampling_parameters['sampling_traj_interval']
            self.current_step, self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            
            
    def get_current_structure(self):
        if self.sampling_parameters['sampling_initialization']:
            system = read_xyz_file('MoREST.str')
        else:
            system = self.current_traj[-1]
            #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            
        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses
        
        if self.sampling_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_FD_forces(system, \
                                                      self.sampling_parameters['fd_displacement'])
        else:
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return self.current_step, system
    

class velocity_Verlet(initialize_sampling):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling, and velocity rescaling method to constrain the kinetic energy in a NVT MD system.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str records the initial xyz structure of the system
    MoREST.str_new records the current xyz structure of the system
    '''
    
    def __init__(self, sampling_parameters, md_parameters, calculator=None, v_rescaling=False, sv_rescaling=False):
        super(velocity_Verlet, self).__init__(sampling_parameters, md_parameters, calculator)
        self.v_rescaling = v_rescaling
        self.sv_rescaling = sv_rescaling
        
        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.md_parameters['md_temperature'] # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        
        if self.v_rescaling:
            self.velocity_rescaling(self.current_system)
        
        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open('MoREST_MD.log', 'w', buffering=1)
            if self.sv_rescaling:
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, self.K_simulation, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            else:
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
                write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open('MoREST_MD.log', 'a', buffering=1)
            if self.sv_rescaling:
                self.d_Ee = 0
                self.Wt =  0
            #else:
            #    self.MD_log = open('MoREST_MD.log', 'a', buffering=1)
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.md_parameters['md_time_step']
        
        next_system = deepcopy(self.current_system)
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            self.current_forces = self.current_forces + bias_forces
        
        ### x(t), v(t) = p(t) / m
        current_coordinates = self.current_system.get_positions()
        #current_velocities = self.current_system.get_velocities()
        current_momenta = self.current_system.get_momenta()
        
        ### x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt^2/m
        #next_coordinates = current_coordinates + current_velocities * time_step + 0.5 * self.current_accelerations * time_step**2
        next_coordinates = current_coordinates + (current_momenta * time_step + 0.5 * self.current_forces * time_step**2) / self.masses
        next_system.set_positions(next_coordinates)
        
        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * self.current_forces * time_step
        
        ### F(t+dt)
        if self.sampling_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            next_potential_energy, next_forces = self.many_body_potential.get_potential_FD_forces(next_system, \
                                                      self.sampling_parameters['fd_displacement'])
        else:
            next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(next_system)
        
        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        #next_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        next_system.set_momenta(next_momenta)
        
        next_velocities = next_system.get_velocities()
        
        self.current_step += 1
        self.current_system = next_system
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        
        if self.v_rescaling:
            self.velocity_rescaling()
        
        if self.sv_rescaling:
            R_t = self.stochastic_velocity_rescaling()
        
        if self.md_parameters['md_clean_translation']:
            #next_velocities = clean_translation(next_velocities)
            Stationary(self.current_system)
        if self.md_parameters['md_clean_rotation']:
            #next_velocities = clean_rotation(next_velocities, next_coordinates, self.masses)
            ZeroRotation(self.current_system)
        
        write_xyz_file('MoREST.str_new', self.current_system)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            self.current_traj.append(self.current_system)
            write_xyz_traj('MoREST_traj.xyz', self.current_system)
            kinetic_energy = self.current_system.get_kinetic_energy()
            if self.sv_rescaling:
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses, self.K_simulation, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, self.Wt+R_t)
            else:
                write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system
    
    def velocity_rescaling(self):
        dT = self.sampling_parameters['nvt_vr_dt']
        lower_T = self.md_parameters['md_temperature'] - dT
        upper_T = self.md_parameters['md_temperature'] + dT
        Ek = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        if Ti > upper_T or Ti < lower_T:
            factor = np.sqrt(self.md_parameters['md_temperature'] / Ti)
            self.current_system.set_velocities(factor * velocities)
        
    def stochastic_velocity_rescaling(self):
        '''
        This function implements stochastic velocity rescaling algorithm (Bussi, Donadio and Parrinello, JCP (2007); Bussi, Parrinello, CPC (2008)) to do canonical ensenmble sampling (NVT MD).
        '''
        tau = self.sampling_parameters['nvt_svr_tau']
        time_step = self.md_parameters['md_time_step']
        
        ### degree of freedom
        Nf = 3 * self.n_atom
        #if self.sampling_parameters['sampling_clean_translation']:
        #    Nf = Nf - 3
        #if self.sampling_parameters['sampling_clean_rotation']:
        #Nf = Nf - 3
            
        ### Gaussian random number R(t)
        R = np.random.normal(size=Nf)
        R_t = R[0]
        S_Nf_1 = np.sum(R[1:]**2)
        
        ### c = exp(- time_step / tau)
        c = np.exp(-1 * time_step / tau )
        
        ### kinetic energy K
        K_t = self.current_system.get_kinetic_energy()
        factor = self.K_simulation / K_t / Nf
        
        ### alpha
        alpha = np.sqrt(c + (1-c)*(S_Nf_1 + R_t**2)*factor + 2*R_t*np.sqrt(c*(1-c)*factor))
        sign = np.sign(R_t + np.sqrt(c/(1-c)/factor))
        alpha = alpha * sign
        
        velocities = self.current_system.get_velocities()
        self.current_system.set_velocities(alpha * velocities)
        
        return R_t
    
        
def clean_translation(velocities):
    total_velocity = np.sum(velocities, axis=0)/len(velocities)
    velocities = velocities - total_velocity
    return velocities
    
def clean_rotation(velocities, coordinates, masses):
    v_vector = velocities
    #center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
    center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    r_vector = coordinates - center_of_mass
        
    r_cross_v = np.cross(r_vector, v_vector)
    r_2 = np.linalg.norm(r_vector, axis=1)**2
    omega = np.array([r_cross_v[i]/r_2[i] for i in range(4)])
    rotat_vector = np.sum(omega, axis=0)/len(masses)
    v_tang = np.cross(rotat_vector, r_vector)
    velocities = v_vector - v_tang
        
    return velocities
        
def write_MD_log(MD_log, step, Ep, Ek, masses):
    n_atom = len(masses)
    #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
    #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
    T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    Et = Ek + Ep
    MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
    
def write_SVR_MD_log(MD_log, step, Ep, Ek, masses, K_simulation, tau, d_Ee, Wt):
    n_atom = len(masses)
    Nf = 3 * n_atom
    #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
    #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
    T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    Et = Ek + Ep
    d_Ee = d_Ee -1*((K_simulation - Ek)/tau + 2*np.sqrt(Ek*K_simulation/Nf/tau)*Wt)
    Ee = Et + d_Ee
    MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
    return d_Ee, Wt
    
