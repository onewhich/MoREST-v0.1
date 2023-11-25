#from time import time
import numpy as np
#import sys
#sys.path.append('..')
from structure_io import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from initialization import initialize_calculator
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, stochastic_velocity_rescaling
from barostat import barostat_space, Berendsen_volume_rescaling, stochastic_velocity_volume_rescaling

class initialize_sampling(initialize_calculator):
    def __init__(self, morest_parameters, sampling_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):
        super(initialize_sampling, self).__init__(morest_parameters, calculator, log_morest)
        self.sampling_parameters = sampling_parameters
        if traj_file_name == None:
            self.traj_file_name = 'MoREST_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        if self.sampling_parameters['sampling_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
        else:
            try:
                self.current_traj = read_xyz_traj(self.traj_file_name)
                self.current_step = (len(self.current_traj) - 1) * self.sampling_parameters['sampling_traj_interval']
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.current_step = 0
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure(molecule)
            
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
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
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
        
        return system
    
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
        
    @staticmethod
    def write_SVR_MD_log(MD_log, step, Ep, Ek, masses, Ee=0, d_Ee=0):
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
        Ee += d_Ee
        MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
        return Ee
    
    @staticmethod
    def write_SVR_MD_log_old(MD_log, step, Ep, Ek, masses, K_simulation, time_step, tau, d_Ee, Wt):
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
        except:
            pass
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
    
    @staticmethod
    def clean_translation(velocities):
        total_velocity = np.sum(velocities, axis=0)/len(velocities)
        velocities = velocities - total_velocity
        
        return velocities
        
    @staticmethod
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
            

class velocity_Verlet(initialize_sampling):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling, and (stochestic) velocity rescaling method to constrain the kinetic energy in a NVT MD system.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str (default name) records the initial xyz structure of the system
    MoREST.str_new (default name) records the current xyz structure of the system
    '''
    
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super(velocity_Verlet, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
        self.md_parameters = md_parameters
        self.time_step = self.md_parameters['md_time_step']
        
        if T_simulation == None:
            self.re_simulation = False
            self.T_simulation = self.md_parameters['md_temperature']
        else:
            self.re_simulation = True
            self.T_simulation = T_simulation

        if self.current_step == 0:
            if self.sampling_parameters['sampling_pre_thermalized']:
                if 'sampling_initial_E' in self.sampling_parameters:
                    T_thermalized = 2/3 * self.sampling_parameters['sampling_initial_E']/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif 'sampling_initial_T' in self.sampling_parameters:
                    T_thermalized = self.sampling_parameters['sampling_initial_T']
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized)
                    self.pre_thermalization(T_thermalized)
                elif self.T_simulation > 1e-3:
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)
            
            #self.current_traj = []
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
        
        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.T_simulation # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        
    def VV_next_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        if time_step == None:
            time_step = self.time_step

        if updated_current_system != None:
            self.current_system = updated_current_system
        
        ### F(t) + bias
        if bias_forces != None:
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
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * self.time_step
        #self.current_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        self.current_system.set_momenta(next_momenta)
        
        #next_velocities = next_system.get_velocities()
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

    def pre_thermalization(self, Tf):
        Ek_i = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek_i/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        velocities = self.current_system.get_velocities()
        factor = np.sqrt(Tf / Ti)
        self.current_system.set_velocities(factor * velocities)

    @staticmethod
    def get_temperature(Ek, n_atom):
        return 2/3 * Ek/units.kB /n_atom


class NVE_VV(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system

class NVK_VR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NVT_Berendsen(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NVT_Langevin(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), 0, 0)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  1, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), self.d_Ee, R_t)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NVT_SVR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NPH_SVR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.NPH_space = barostat_space(md_parameters, self.current_system)

        self.P_simulation = self.md_parameters['barostat_pressure']
        self.tau_P = self.sampling_parameters['nph_svr_tau']
        self.eta = np.zeros(self.md_parameters['barostat_number']) # initialize the velocity of the barostat
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf = 3*self.n_atom - 2
        self.half_time_step = self.time_step/2
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        self.W_barostat = self.Nf * units.kB * T_current * self.tau_P**2

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        
        NPT_bias_forces = self.NPH_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        self.VV_next_step(bias_forces, updated_current_system)

        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        # stage 2: propagate 1/2 time step velocities
        # stage 3: propagate time step positions and velocities
        # stage 4: propagate 1/2 time step velocities
        new_coordinates, new_momenta, self.eta, P_current = stochastic_velocity_volume_rescaling(self.md_parameters, self.time_step, self.half_time_step, \
                                                            self.current_system.get_positions(), self.current_system.get_forces(), self.current_system.get_velocities(), \
                                                            self.eta, self.current_system.get_momenta(), self.masses, self.W_barostat, T_current, self.P_simulation)
        self.current_system.set_positions(new_coordinates)
        self.current_system.set_momenta(new_momenta)
        self.NPH_space.update_barostat_space_wall()
        
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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NPT_Berendsen(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.NPT_space = barostat_space(md_parameters, self.current_system)

        self.P_simulation = self.md_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_Berendsen_tau_t']
        self.tau_P = self.sampling_parameters['npt_Berendsen_tau_p']
        self.beta = self.sampling_parameters['npt_Berendsen_compressibility']
        init_miu = np.ones(self.md_parameters['barostat_number']) # the first rescaling factor should be one for each barostat space

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        next_coordinates, self.miu, P_current = Berendsen_volume_rescaling(self.md_parameters, self.time_step, self.current_system.get_positions(), \
                                                               self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, self.beta, init_miu)
        self.current_system.set_positions(next_coordinates)
        self.NPT_space.update_barostat_space_wall()

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        self.VV_next_step(bias_forces, updated_current_system)
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        next_coordinates, self.miu, P_current = Berendsen_volume_rescaling(self.md_parameters, self.time_step, self.current_system.get_positions(), \
                                                               self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, self.beta, self.miu)
        self.current_system.set_positions(next_coordinates)
        self.NPT_space.update_barostat_space_wall()

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NPT_Langevin(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.NPT_space = barostat_space(md_parameters, self.current_system)


class NPT_SVR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.NPT_space = barostat_space(md_parameters, self.current_system)

        self.P_simulation = self.md_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_svr_tau_t']
        self.tau_P = self.sampling_parameters['npt_svr_tau_p']
        self.eta = np.zeros(self.md_parameters['barostat_number']) # initialize the velocity of the barostat
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf = 3*self.n_atom - 2
        self.half_time_step = self.time_step/2
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        self.W_barostat = self.Nf * units.kB * T_current * self.tau_P**2

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        self.VV_next_step(bias_forces, updated_current_system)

        # stage 1: propagate 1/2 time step thermostat
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step/2, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  self.Nf, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        self.eta *= alpha
        
        # stage 2: propagate 1/2 time step velocities
        # stage 3: propagate time step positions and velocities
        # stage 4: propagate 1/2 time step velocities
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        new_coordinates, new_momenta, self.eta, P_current = stochastic_velocity_volume_rescaling(self.md_parameters, self.time_step, self.half_time_step, \
                                                            self.current_system.get_positions(), self.current_system.get_forces(), new_velocities, self.eta, self.current_system.get_momenta(), \
                                                            self.masses, self.W_barostat, T_current, self.P_simulation)
        self.current_system.set_positions(new_coordinates)
        self.current_system.set_momenta(new_momenta)
        self.NPT_space.update_barostat_space_wall()

        # stage 5: propagate 1/2 time step thermostat
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step/2, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  self.Nf, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        self.eta *= alpha

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
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system
