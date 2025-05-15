import os
from copy import deepcopy
import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_file, write_xyz_traj
from initialize_calculator import initialize_calculator
from numerical_integraion import MD_integration, RPMD_integration, RPMD_normal_mode_integration
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
from kinetic_energy_assignment import clean_translation, clean_rotation
from ase import units

class initialize_sampling(initialize_calculator):
    def __init__(self, morest_parameters, sampling_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):
        super(initialize_sampling, self).__init__(morest_parameters, calculator, log_morest)
        self.sampling_parameters = sampling_parameters

        if self.sampling_parameters['sampling_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
        else:
            try:
                self.current_traj = read_xyz_traj(traj_file_name)
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
            if type(molecule) == type(None):
                system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
                if type(molecule) == type(None):
                    system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses

        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return system
    
    @staticmethod
    def write_MD_log(log_file, step, Ep, Ek, masses):
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
        log_file.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
        
    @staticmethod
    def write_MD_SVR_log(log_file, step, Ep, Ek, masses, Ee=0, d_Ee=0):
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
        log_file.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
        return Ee
    
    @staticmethod
    def write_MD_SVR_log_old(log_file, step, Ep, Ek, masses, K_simulation, time_step, tau, d_Ee, Wt):
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
        log_file.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(Ee)+'\n')
        return d_Ee, Wt
    
class MD(initialize_sampling):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) sampling, and (stochestic) velocity rescaling method to constrain the kinetic energy in a NVT MD system.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.xyz (default name) records the initial xyz structure of the system
    MoREST.xyz_new (default name) records the current xyz structure of the system
    '''
    
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        super(MD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
        self.MD_parameters = MD_parameters
        self.time_step = self.MD_parameters['md_time_step']
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        if type(T_simulation) == type(None):
            self.re_simulation = False
            self.T_simulation = self.MD_parameters['md_temperature']
        else:
            self.re_simulation = True
            self.T_simulation = T_simulation

        if self.current_step == 0:
            if not self.sampling_parameters['sampling_pre_thermalized']:
                if 'sampling_initial_E' in self.sampling_parameters:
                    T_thermalized = 2/3 * self.sampling_parameters['sampling_initial_E']/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized, force_temp = True)
                elif 'sampling_initial_T' in self.sampling_parameters:
                    T_thermalized = self.sampling_parameters['sampling_initial_T']
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized, force_temp = True)
                else:
                    MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)
            
            #self.current_traj = []
            #self.current_traj.append(self.current_system)
            write_xyz_traj(traj_file_name, self.current_system)
        
        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.T_simulation # Ek = 1/2 m v^2 = 3/2 kB T for each particle

        self.integration = MD_integration(self.many_body_potential)

    def update_pre_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        if type(time_step) == type(None):
            time_step = self.time_step

        if type(updated_current_system) != type(None):
            self.current_system = updated_current_system
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            self.current_forces = self.current_forces + bias_forces

        return time_step
        
    def update_step(self, next_potential_energy, next_forces):
        
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

        if self.MD_parameters['md_clean_rotation']:
            clean_rotation(self.current_system, preserve_temperature=True)
        if self.MD_parameters['md_clean_translation']:
            clean_translation(self.current_system, preserve_temperature=True)
            
        self.current_step += 1
        
        if not self.re_simulation:
            write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)
        else:
            write_xyz_file('MoREST_RE_'+str(self.T_simulation)+'K.xyz_new', self.current_system)

    def pre_thermalization(self, Tf):
        Ek_i = self.current_system.get_kinetic_energy()
        Ti = 2/3 * Ek_i/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        velocities = self.current_system.get_velocities()
        factor = np.sqrt(Tf / Ti)
        self.current_system.set_velocities(factor * velocities)

    @staticmethod
    def get_temperature(Ek, n_atom):
        return 2/3 * Ek/units.kB /n_atom

class RPMD(initialize_sampling):
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    '''
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):

        self.beads_file_head = 'MoREST_RPMD_beads_'

        super(RPMD, self).__init__(morest_parameters, sampling_parameters, molecule, traj_file_name, calculator, log_morest)
        self.beta = RPMD_parameters['beta']
        self.hbar = RPMD_parameters['hbar']
        self.n_beads = RPMD_parameters['rpmd_number_of_beads']
        self.beads_file_name = RPMD_parameters['rpmd_beads_file']
        self.time_step = RPMD_parameters['rpmd_time_step']
        self.T_simulation = RPMD_parameters['rpmd_temperature']
        self.omega_n = RPMD_parameters['omega_n']
        self.omega_k = RPMD_parameters['omega_k']
        self.C_jk = RPMD_parameters['C_jk']
        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']
        self.atom_masses = self.masses.flatten()
        self.current_system.calc = calculator

        ### kinetic energy at simulation temperature
        Nf = 3 * self.n_atom
        self.K_simulation = Nf/2 * units.kB * self.T_simulation # Ek = 1/2 m v^2 = 3/2 kB T for each particle

        if os.path.isfile(self.beads_file_name):
            self.current_beads = read_xyz_traj(self.beads_file_name)
            log_morest.write('Read beads from file: '+self.beads_file_name+'\n\n')
            if len(self.current_beads) != self.n_beads:
                raise Exception('The number of structures in beads file does not fit the number of beads given by the parameter file. Please check.')
        else:
            self.initialize_beads_noraml_mode()
            log_morest.write('Initialize beads and write to file: '+self.beads_file_name+'\n\n')
        write_xyz_file(self.beads_file_name, self.current_beads)
        write_xyz_file('MoREST_RPMD_beads_initial.xyz', self.current_beads)

        if self.current_step == 0:
            if not self.sampling_parameters['sampling_pre_thermalized']:
                if 'sampling_initial_E' in self.sampling_parameters:
                    T_thermalized = 2/3 * self.sampling_parameters['sampling_initial_E']/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_beads[i], temperature_K = T_thermalized, force_temp = True)
                elif 'sampling_initial_T' in self.sampling_parameters:
                    T_thermalized = self.sampling_parameters['sampling_initial_T']
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_beads[i], temperature_K = T_thermalized, force_temp = True)
                else:
                    for i in range(self.n_beads):
                        MaxwellBoltzmannDistribution(self.current_system, temperature_K = self.T_simulation)


        self.current_beads_positions = self.get_beads_positions(self.current_beads)
        self.current_beads_momenta = self.get_beads_momenta(self.current_beads)
        self.current_beads_potential_energy, self.current_beads_forces = self.get_beads_potential_forces(self.current_beads)

        self.update_centroid_positions_momenta(self.current_beads)
        self.update_centroid_potential_energy_forces(self.current_beads_potential_energy, self.current_beads_forces)

        self.integration = RPMD_integration(self.many_body_potential, self.omega_n, self.n_beads)

    def initialize_beads_noraml_mode(self):
        centroid_positions = self.current_system.get_positions().reshape((1, self.n_atom, 3))
        masses = self.masses.reshape((1, self.n_atom, 1))
        self.current_beads = []
        for i in range(self.n_beads):
            tmp_system = deepcopy(self.current_system)
            self.current_beads.append(tmp_system)
        rng = np.random.default_rng()
        
        # Initialize normal mode displacements
        positions_normal_mode = np.zeros((self.n_beads, self.n_atom, 3))

        for k_mode in range(1, self.n_beads):  # skip k = 0 (centroid)
            sigma = np.sqrt(1.0 / (masses * self.beta * self.omega_k[k_mode] ** 2))
            positions_normal_mode[k_mode] = rng.normal(0.0, sigma, size=(1, self.n_atom, 3))

        # Inverse Fourier transform (real positions)
        self.current_beads_positions = np.fft.ifft(positions_normal_mode, axis=0).real * self.n_beads

        # Shift to match centroids
        self.current_beads_positions += centroid_positions - np.mean(self.current_beads_positions, axis=0, keepdims=True)

        self.update_beads_positions(self.current_beads_positions)

    def initialize_beads_real_space(self, factor_r=0.7):
        # r_beads: the average distance from a bead to the neighbor for free particles.
        r_beads = [np.sqrt(self.beta * (self.hbar)**2 / self.n_beads / self.atom_masses[i]) for i in range(self.n_atom)] 
        
        # r_ring: the radius of the beads gyration in imaginary time.
        lambda_T = np.array([units._hplanck*units.J*units.s / np.sqrt(2*np.pi*self.atom_masses[i]*units.kB*self.T_simulation) for i in range(self.n_atom)])[:,np.newaxis]
        r_ring = lambda_T / np.sqrt(8*np.pi)

        centroid_pos = self.current_system.get_positions()
        self.current_beads = []

        # the position of the first bead
        # every bead stays r_ring away from the centroid.
        rand_pos_1 = np.random.rand(self.n_atom, 3) - 0.5
        norm_1 = np.linalg.norm(rand_pos_1,axis=-1)[:,np.newaxis]
        pos_new_1 = rand_pos_1/norm_1*r_ring + centroid_pos
        tmp_system = deepcopy(self.current_system)
        tmp_system.set_positions(pos_new_1)
        self.current_beads.append(tmp_system)

        # the positions of other beads
        while len(self.current_beads) < self.n_beads:
            beads_pos = np.array([i_bead.get_positions() for i_bead in self.current_beads])
            tmp_pos = []
            i = 0
            while len(tmp_pos) < self.n_atom:
                atoms_pos = beads_pos[:,i,:]

                rand_pos = np.random.rand(3) - 0.5
                norm = np.linalg.norm(rand_pos,axis=-1)
                pos_new = np.array(rand_pos/norm*r_ring[i] + centroid_pos[i])

                # each bead is farther from every other bead than r_beads times a factor.
                if np.all(np.linalg.norm(atoms_pos - pos_new,axis=-1) > (factor_r * r_beads[i])):
                    tmp_pos.append(pos_new)
                    i += 1
            tmp_system = deepcopy(self.current_system)
            tmp_system.set_positions(tmp_pos)
            self.current_beads.append(tmp_system)

    def RPMD_update_pre_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        return time_step

    def RPMD_update_step(self, current_beads_potential_energy, current_beads_forces, next_beads_positions, next_beads_momenta):

        self.update_beads_positions(next_beads_positions)
        self.current_beads_positions = next_beads_positions
        self.update_beads_momenta(next_beads_momenta)
        self.current_beads_momenta = next_beads_momenta
        self.update_centroid_positions_momenta(self.current_beads)
        self.update_centroid_potential_energy_forces(current_beads_potential_energy, current_beads_forces)
        
        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        self.current_step += 1

        write_xyz_file(self.beads_file_name, self.current_beads)
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

    def pre_thermalization(self, Tf):
        for i in range(self.n_beads):
            Ek_i = self.current_beads[i].get_kinetic_energy()
            Ti = 2/3 * Ek_i/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
            velocities = self.current_beads[i].get_velocities()
            factor = np.sqrt(Tf / Ti)
            self.current_beads[i].set_velocities(factor * velocities)
            
    def get_beads_positions(self, beads):
        beads_positions = np.array([i_bead.get_positions() for i_bead in beads])
        return beads_positions
    
    def update_beads_positions(self, new_beads_positions):
        for i in range(self.n_beads):
            self.current_beads[i].set_positions(new_beads_positions[i])

    def get_beads_momenta(self, beads):
        beads_momenta = np.array([i_bead.get_momenta() for i_bead in beads])
        return beads_momenta
    
    def update_beads_momenta(self, new_beads_momenta):
        for i in range(self.n_beads):
            self.current_beads[i].set_momenta(new_beads_momenta[i])

    def get_beads_kinetic_energy(self, beads):
        beads_kinetic_energy = [i_bead.get_kinetic_energy() for i_bead in beads]
        return np.array(beads_kinetic_energy)
            
    def get_beads_potential_forces(self, beads):
        beads_potential_energy = []
        beads_forces = []
        for i in range(self.n_beads):
            tmp_potential_energy, tmp_forces = self.many_body_potential.get_potential_forces(beads[i])
            beads_potential_energy.append(tmp_potential_energy)
            beads_forces.append(tmp_forces)
        return np.array(beads_potential_energy), np.array(beads_forces)
    
    def update_centroid_positions_momenta(self, beads):
        beads_positions = self.get_beads_positions(beads)
        system_positions = np.average(beads_positions, axis=0)
        self.current_system.set_positions(system_positions)
        beads_momenta = self.get_beads_momenta(beads)
        system_momenta = np.average(beads_momenta, axis=0)
        self.current_system.set_momenta(system_momenta)

    def update_centroid_potential_energy_forces(self, beads_potential_energy, beads_forces):
        self.current_system.calc.results['energy'] = np.average(beads_potential_energy)
        self.current_system.calc.results['forces'] = np.average(beads_forces, axis=0)

    # only remove the centroid translational motion
    def clean_translation_centroid(self):
        centroid_velocity = self.current_system.get_velocities()
        clean_translation(self.current_system, preserve_temperature=True)
        new_velocity = self.current_system.get_velocities()
        d_velocity = new_velocity - centroid_velocity
        for i_bead in self.current_beads:
            bead_velocity = i_bead.get_velocities()
            i_bead.set_velocities(bead_velocity + d_velocity)

    # only remove the centroid rotational motion
    def clean_rotation_centroid(self):
        centroid_velocity = self.current_system.get_velocities()
        clean_rotation(self.current_system, preserve_temperature=True)
        new_velocity = self.current_system.get_velocities()
        d_velocity = new_velocity - centroid_velocity
        for i_bead in self.current_beads:
            bead_velocity = i_bead.get_velocities()
            i_bead.set_velocities(bead_velocity + d_velocity)

class RPMD_normal_mode(RPMD):
    '''
    The ring polymer molecular dynamics in normal mode representation.
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, traj_file_name=None, calculator=None, log_morest=None):
        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)
        self.integration = RPMD_normal_mode_integration(self.many_body_potential)