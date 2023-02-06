from time import time
import numpy as np
#import sys
#sys.path.append('..')
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_interface, on_the_fly, molpro_calculator
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units

class initialize_sampling:
    def __init__(self, morest_parameters, sampling_parameters, calculator=None, log_file=None):
        self.morest_parameters = morest_parameters
        self.sampling_parameters = sampling_parameters
        
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
            ml_calculator = ml_interface(ab_initio_calculator = calculator, \
                                    ml_parameters = self.morest_parameters, \
                                    log_file = log_file)
            self.many_body_potential = on_the_fly(ml_calculator)
            
        else:
            raise Exception('Which many body potential will you use?')
            
    def get_current_structure(self, molecule=None):
        if self.sampling_parameters['sampling_initialization']:
            if molecule == None:
                system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                if molecule == None:
                    system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses
        
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)

      

        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return self.current_step, system
    

class velocity_Verlet(initialize_sampling):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling, and (stochestic) velocity rescaling method to constrain the kinetic energy in a NVT MD system.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str (default name) records the initial xyz structure of the system
    MoREST.str_new (default name) records the current xyz structure of the system
    '''
    
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, \
                        v_rescaling=False, Berendsen_rescaling=False, Langevin_rescaling=False, sv_rescaling=False, log_file=None):
        super(velocity_Verlet, self).__init__(morest_parameters, sampling_parameters, calculator, log_file)
        self.md_parameters = md_parameters
        self.traj_file_name = traj_file_name
        self.log_file_name = log_file_name
        self.v_rescaling = v_rescaling
        self.b_rescaling = Berendsen_rescaling
        self.l_rescaling = Langevin_rescaling
        self.sv_rescaling = sv_rescaling
        if T_simulation == None:
            self.re_simulation = False
            self.T_simulation = self.md_parameters['md_temperature']
        else:
            self.re_simulation = True
            self.T_simulation = T_simulation
        
        if self.sampling_parameters['sampling_initialization']:
            self.current_step = 0
            self.current_step, self.current_system = self.get_current_structure(molecule)
            if self.T_simulation > 1e-6:
                MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)
            self.current_traj = []
            self.current_traj.append(self.current_system)
            if self.traj_file_name == None:
                write_xyz_traj('MoREST_traj.xyz', self.current_system)
            else:
                write_xyz_traj(self.traj_file_name, self.current_system)
        else:
            try:
                if self.traj_file_name == None:
                    self.current_traj = read_xyz_traj('MoREST_traj.xyz')
                else:
                    self.current_traj = read_xyz_traj(self.traj_file_name)
                self.current_step = (len(self.current_traj) - 1) * self.sampling_parameters['sampling_traj_interval']
                self.current_step, self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.current_step = 0
                self.current_step, self.current_system = self.get_current_structure(molecule)
                if self.T_simulation > 1e-6:
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)
                self.current_traj = []
                self.current_traj.append(self.current_system)
                if self.traj_file_name == None:
                    write_xyz_traj('MoREST_traj.xyz', self.current_system)
                else:
                    write_xyz_traj(self.traj_file_name, self.current_system)

        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.T_simulation # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        
        if self.v_rescaling:
            self.velocity_rescaling()
        elif self.b_rescaling:
            self.Berendsen_rescaling()
        
        if self.sampling_parameters['sampling_initialization']:
            if self.log_file_name == None:
                self.MD_log = open('MoREST_MD.log', 'w', buffering=1)
            else:
                self.MD_log = open(self.log_file_name, 'w', buffering=1)
            if self.l_rescaling:
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, self.K_simulation, self.md_parameters['md_time_step'], 1/(2*self.sampling_parameters['nvt_langevin_gamma']), 0, 0)
            elif self.sv_rescaling:
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, self.K_simulation, self.md_parameters['md_time_step'], self.sampling_parameters['nvt_svr_tau'], 0, 0)
            else:
                self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
                write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            if self.log_file_name == None:
                self.MD_log = open('MoREST_MD.log', 'a', buffering=1)
            else:
                self.MD_log = open(self.log_file_name, 'a', buffering=1)
            if self.sv_rescaling:
                self.d_Ee = 0
                self.Wt =  0
        
    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        time_step = self.md_parameters['md_time_step']
        
        if not updated_current_system == None:
            self.current_system = updated_current_system
        
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
        self.current_system.set_positions(next_coordinates)
        
        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * self.current_forces * time_step
        
        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        #self.current_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        self.current_system.set_momenta(next_momenta)
        
        #next_velocities = next_system.get_velocities()
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        
        if self.v_rescaling:
            self.velocity_rescaling()
        elif self.b_rescaling:
            self.Berendsen_rescaling()
        elif self.l_rescaling:
            R_t = self.stochastic_velocity_rescaling(Nf = 1, tau = 1/(2*self.sampling_parameters['nvt_langevin_gamma']))
        elif self.sv_rescaling:
            R_t = self.stochastic_velocity_rescaling(Nf = 3*self.n_atom, tau = self.sampling_parameters['nvt_svr_tau'])
        
        if self.md_parameters['md_clean_translation']:
            #next_velocities = clean_translation(next_velocities)
            Stationary(self.current_system)
        if self.md_parameters['md_clean_rotation']:
            #next_velocities = clean_rotation(next_velocities, next_coordinates, self.masses)
            ZeroRotation(self.current_system)
        
        if not self.re_simulation:
            write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)
        else:
            write_xyz_file('MoREST_RE_'+str(self.T_simulation)+'K.str_new', self.current_system)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            self.current_traj.append(self.current_system)
            if self.traj_file_name == None:
                write_xyz_traj('MoREST_traj.xyz', self.current_system)
            else:
                write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            if self.l_rescaling:
                #self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses, self.K_simulation, time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, self.Wt+R_t)
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.K_simulation, time_step, 1/(2*self.sampling_parameters['nvt_langevin_gamma']), self.d_Ee, R_t)
            elif self.sv_rescaling:
                #self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses, self.K_simulation, time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, self.Wt+R_t)
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.K_simulation, time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            else:
                write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system
    
    def velocity_rescaling(self):
        dT = self.sampling_parameters['nvt_vr_dt']
        lower_T = self.T_simulation - dT
        upper_T = self.T_simulation + dT
        Ek = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        velocities = self.current_system.get_velocities()
        if Ti > upper_T or Ti < lower_T:
            factor = np.sqrt(self.T_simulation / Ti)
            self.current_system.set_velocities(factor * velocities)

    def Berendsen_rescaling(self):
        tau = self.sampling_parameters['nvt_berendsen_tau']
        time_step = self.md_parameters['md_time_step']
        Ek = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        factor = np.sqrt(1 + time_step/tau * (self.T_simulation/Ti -1))
        velocities = self.current_system.get_velocities()
        self.current_system.set_velocities(factor * velocities)
        
    def stochastic_velocity_rescaling(self, Nf, tau):
        '''
        This function implements stochastic velocity rescaling algorithm (Bussi, Donadio and Parrinello, JCP (2007); Bussi, Parrinello, CPC (2008)) to do canonical ensenmble sampling (NVT MD).
        '''
        time_step = self.md_parameters['md_time_step']
        
        ### degree of freedom
        # Nf = 1                # for Langevin thermostat
        # Nf = 3 * self.n_atom  # for SVR thermostat
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
    '''
    L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
    L : angular momentum
    omega: angular velocity
    I : moment of inertia
    '''
    v_vector = velocities
    #center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
    center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    r_vector = coordinates - center_of_mass
    # r_cross_v : angular velocities
    # omega = (r x v) / |r|^2
    r_cross_v = np.cross(r_vector, v_vector)
    r_2 = np.linalg.norm(r_vector, axis=1)**2
    omega = np.array([r_cross_v[i]/r_2[i] for i in range(4)])
    # Rv = omega/n_atom : system total angular velocity
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
    
def write_SVR_MD_log(MD_log, step, Ep, Ek, masses, K_simulation, time_step, tau, d_Ee, Wt):
    n_atom = len(masses)
    Nf = 3 * n_atom
    #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
    #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
    T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    Et = Ek + Ep
    d_Ee = d_Ee + (K_simulation - Ek)*time_step/tau + 2*np.sqrt(Ek*K_simulation/Nf/tau)*Wt
    Ee = Et - d_Ee
    MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
    return d_Ee, Wt
    
