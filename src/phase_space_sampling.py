#from time import time
import numpy as np
#import sys
#sys.path.append('..')
from structure_io import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from initialization import initialize_calculator
#from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from wall_potential import repulsive_wall

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
        return Ee, d_Ee
    
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
        
    def VV_next_step(self, bias_forces=None, updated_current_system=None):
        if type(updated_current_system) != type(None):
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
        next_coordinates = current_coordinates + (current_momenta * self.time_step + 0.5 * self.current_forces * self.time_step**2) / self.masses
        self.current_system.set_positions(next_coordinates)
        
        ### v(t+0.5dt) = p(t+0.5dt) / m; p(t+0.5dt) = p(t) + 0.5 * F(t) * dt
        momenta_half = current_momenta + 0.5 * self.current_forces * self.time_step
        
        ### F(t+dt)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * self.time_step
        #self.current_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * self.time_step
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

    def Berendsen_velocity_rescaling(self, tau):
        time_step = self.md_parameters['md_time_step']
        Ek = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        factor = np.sqrt(1 + time_step/tau * (self.T_simulation/Ti -1))
        velocities = self.current_system.get_velocities()
        self.current_system.set_velocities(factor * velocities)

    def Berendsen_volume_rescaling(self, tau_P, beta, factor):
        time_step = self.md_parameters['md_time_step']
        Eks = self.get_atom_kinetic_energies()
        next_factor = []
        forces = self.current_forces
        coordinates = self.current_system.get_positions()
        for i in range(self.md_parameters['npt_number']):
            index = self.md_parameters['npt_action_atoms'][i]
            if index == 'all':
                index = np.arange(self.n_atom)
            internal_virial = self.get_internal_virial(coordinates[index], forces[index])
            current_pressure = (Eks[i] - internal_virial)*2/(3.*self.get_volume(i))
            tmp_factor = np.power(1+(time_step/tau_P)*beta*(current_pressure-self.P_simulation),1./3.)
            next_factor.append(tmp_factor)

            if self.md_parameters['npt_space_type'][i].upper() == 'gas'.upper():
                self.md_parameters['npt_space_size'][i] *= factor[i]
            elif self.md_parameters['npt_space_type'][i].upper() == 'condensed'.upper():
                self.md_parameters['npt_space_size'][i] *= factor[i]
                coordinates[index] *= factor[i]

        self.current_system.set_positions(coordinates)

        return np.array(next_factor)
        
    def stochastic_velocity_rescaling(self, Nf, tau):
        '''
        This function implements stochastic velocity rescaling algorithm (Bussi, Donadio and Parrinello, JCP (2007); Bussi, Parrinello, CPC (2008)) to do canonical ensenmble sampling (NVT MD).
        '''
        time_step = self.md_parameters['md_time_step']
        
        ### degree of freedom
        # Nf = 1                # for Langevin thermostat
        # Nf = 3 * self.n_atom  # for SVR thermostat
        #if self.md_parameters['md_clean_translation']:
        #    Nf = Nf - 3
        #if self.md_parameters['md_clean_rotation']:
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
        
        return K_t*(1-alpha**2)
    
    def initialize_NPT_space_size(self):
        Eks = self.get_atom_kinetic_energies()
        forces = self.current_forces
        coordinates = self.current_system.get_positions()
        for i in range(self.md_parameters['npt_number']):
            index = self.md_parameters['npt_action_atoms'][i]
            internal_virial = self.get_internal_virial(coordinates[index], forces[index])
            volume = (Eks[i] - internal_virial)/(3.*self.P_simulation)
            if self.md_parameters['npt_space_shape'][i].upper() == 'sphere'.upper():
                self.md_parameters['npt_space_size'].append(np.pow((3/4 * volume / np.pi), 1./3.))  # V = 4/3 * Pi * r^3; r = (3/4 * V/Pi)^(1/3)
        return self.md_parameters['npt_space_size']
    
    def get_internal_virial(self, coordinates, forces):
        return -np.sum([(coordinates[i]-coordinates[j]) @ forces[i] for i in range(self.n_atom-1) for j in range(i+1,self.n_atom)])/2
    
    def get_volume(self, i_space):
        if self.md_parameters['npt_space_shape'][i_space].upper() == 'sphere'.upper():
            volume =  4./3. * np.pi * self.md_parameters['npt_space_size'][i_space]**3 # V = 4/3 * Pi * r^3
        elif self.md_parameters['npt_space_shape'][i_space].upper() == 'cuboid'.upper():
            raise Exception('Cuboid space has not been implemented yet.')
        elif self.md_parameters['npt_space_shape'][i_space].upper() == 'plane'.upper():
            raise Exception('Planar space has not been implemented yet.')
        return volume

    def get_atom_kinetic_energies(self):
        v = np.linalg.norm(self.current_system.get_velocities(), axis=-1)[:,np.newaxis]
        Eks = 0.5 * self.masses * v**2
        return Eks
    
    def initialize_NPT_space_wall(self):
        self.npt_space_wall_parameters = {}
        self.npt_space_wall_parameters['wall_number'] = 0
        self.npt_space_wall_parameters['wall_collective_variable'] = []
        self.npt_space_wall_parameters['wall_shape'] = []
        self.npt_space_wall_parameters['wall_type'] = []
        self.npt_space_wall_parameters['power_wall_direction'] = []
        self.npt_space_wall_parameters['wall_scaling'] = []
        self.npt_space_wall_parameters['wall_scope'] = []
        self.npt_space_wall_parameters['wall_action_atoms'] = []
        self.npt_space_wall_parameters['wall_shape_parameters'] = []
        for i, npt_space in enumerate(self.md_parameters['npt_space_parameters']):
            if self.md_parameters['npt_space_shape'][i].upper() == 'sphere'.upper():
                self.npt_space_wall_parameters['wall_number'] += 1
                self.npt_space_wall_parameters['wall_collective_variable'].append(self.md_parameters['npt_collective_variable'][i])
                self.npt_space_wall_parameters['wall_shape'].append('spherical')
                self.npt_space_wall_parameters['wall_type'].append('power_wall')
                self.npt_space_wall_parameters['power_wall_direction'].append(-1)
                self.npt_space_wall_parameters['wall_scaling'].append(1)
                self.npt_space_wall_parameters['wall_scope'].append(2)
                self.npt_space_wall_parameters['wall_action_atoms'].append(self.md_parameters['npt_number'][i])
                tmp_parameters = {}
                tmp_parameters['spherical_wall_center'] = npt_space['npt_sphere_center']
                tmp_parameters['spherical_wall_radius'] = self.md_parameters['npt_space_size'][i]
                self.npt_space_wall_parameters['wall_shape_parameters'].append(tmp_parameters)
            elif self.md_parameters['npt_space_shape'][i].upper() == 'cuboid'.upper():
                self.npt_space_wall_parameters['wall_number'] += 6
                raise Exception('Cuboidal space has not been implemented yet.')
            elif self.md_parameters['npt_space_shape'][i].upper() == 'plane'.upper():
                self.npt_space_wall_parameters['wall_number'] += 1
                raise Exception('Planar space has not been implemented yet.')
            
        self.NPT_space_wall = repulsive_wall(self.npt_space_wall_parameters)

    def update_NPT_space_wall(self):
        for i, npt_space in enumerate(self.md_parameters['npt_space_parameters']):
            if self.md_parameters['npt_space_shape'][i].upper() == 'sphere'.upper():
                self.npt_space_wall_parameters['wall_shape_parameters'][i]['spherical_wall_radius'] = self.md_parameters['npt_space_size'][i]
        self.NPT_space_wall.update_wall_parameters(self.npt_space_wall_parameters)

    def get_NPT_space_bias_forces(self):
        all_coordinates = self.current_system.get_positions()
        NPT_bias_forces = np.ones((self.md_parameters['npt_number'],3))
        for i in range(self.md_parameters['npt_number']):
            index = self.md_parameters['npt_action_atoms'][i]
            if index == 'all':
                coordinates = all_coordinates
            else:
                coordinates = all_coordinates[index]
            tmp_bias = np.array([self.NPT_space_wall.get_repulsive_wall_force(i_coordinate) for i_coordinate in coordinates])
            for j, j_bias in enumerate(tmp_bias):
                NPT_bias_forces[index[j]] *= j_bias
        return NPT_bias_forces

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

class NVT_VR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.velocity_rescaling()

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.velocity_rescaling()

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

        self.Berendsen_velocity_rescaling(tau = self.sampling_parameters['nvt_berendsen_tau'])

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.Berendsen_velocity_rescaling(tau = self.sampling_parameters['nvt_berendsen_tau'])

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
            #                                           self.masses, self.K_simulation, self.md_parameters['md_time_step'], 1/(2*self.sampling_parameters['nvt_langevin_gamma']), 0, 0)
            self.Ee, self.d_Ee = self.write_SVR_MD_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.d_Ee = self.stochastic_velocity_rescaling(Nf = 1, tau = 1/(2*self.sampling_parameters['nvt_langevin_gamma']))

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
            #                                           self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_langevin_gamma']), self.d_Ee, R_t)
            self.Ee, self.d_Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
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
            #                                           self.masses, self.K_simulation, self.md_parameters['md_time_step'], self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee, self.d_Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.d_Ee = self.stochastic_velocity_rescaling(Nf = 3*self.n_atom, tau = self.sampling_parameters['nvt_svr_tau'])

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
            self.Ee, self.d_Ee = self.write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NPT_Berendsen(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.initialize_NPT_space_size()
        self.initialize_NPT_space_wall()

        self.P_simulation = self.md_parameters['npt_pressure']
        self.tau_T = self.sampling_parameters['npt_Berendsen_tau_t']
        self.tau_P = self.sampling_parameters['npt_Berendsen_tau_p']
        self.beta = self.sampling_parameters['npt_Berendsen_compressibility']
        init_miu = np.ones(self.md_parameters['npt_number']) # the first rescaling factor should be one for each NPT space

        self.Berendsen_velocity_rescaling(tau=self.tau_T)
        self.miu = self.Berendsen_volume_rescaling(tau_P=self.tau_P, beta=self.beta, factor=init_miu)
        self.update_NPT_space_wall()

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        NPT_bias_forces = self.get_NPT_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        self.VV_next_step(bias_forces, updated_current_system)
        self.Berendsen_velocity_rescaling(self.tau_T)
        self.miu = self.Berendsen_volume_rescaling(self.tau_P, self.beta, factor=self.miu)
        self.update_NPT_space_wall()

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

        self.initialize_NPT_space_size()


class NPT_SVR(velocity_Verlet):
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, md_parameters, molecule, traj_file_name, T_simulation, calculator, log_morest)
        if log_file_name == None:
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name

        self.initialize_NPT_space_size()



