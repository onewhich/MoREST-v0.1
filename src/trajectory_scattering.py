from asyncore import write
import numpy as np
from structure import read_xyz_file, write_xyz_file, read_xyz_traj, write_xyz_traj
from many_body_potential import ml_potential, on_the_fly
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units

class initialize_scattering:
    def __init__(self, morest_parameters, scattering_parameters, calculator=None):
        self.morest_parameters = morest_parameters
        self.scattering_parameters = scattering_parameters
        
        if self.morest_parameters['many_body_potential'].upper() in ['on_the_fly'.upper()]:
            if type(calculator) == type(None):
                raise Exception('Please specify the electronic structure method.')
            self.many_body_potential = on_the_fly(calculator)
        elif self.morest_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            trained_ml_potential = self.morest_parameters['ml_potential_model']
            self.many_body_potential = ml_potential(trained_ml_potential)
        else:
            raise Exception('Which many body potential will you use?')

    def generate_scattering_system(self):
        target_molecule = read_xyz_file(self.scattering_parameters['scattering_target_molecule'])
        MaxwellBoltzmannDistribution(target_molecule, temperature_K = self.scattering_parameters['scattering_temperature'])
        Stationary(target_molecule)
        reset_mass_center(target_molecule)

        incident_molecule = read_xyz_file(self.scattering_parameters['scattering_incident_molecule'])
        MaxwellBoltzmannDistribution(incident_molecule, temperature_K = self.scattering_parameters['scattering_temperature'])
        reset_mass_center(incident_molecule)
        # redirect translational movement
        scalar_translational_momentum = np.linalg.norm(get_translational_momentum(incident_molecule))
        target_point = np.random.uniform(-1,1,3)
        target_point = self.scattering_parameters['scattering_R_target'] * target_point / np.linalg.norm(target_point)
        incident_point = np.random.uniform(-1,1,3)
        incident_point = self.scattering_parameters['scattering_R_incident'] * incident_point / np.linalg.norm(incident_point)
        collision_vector = target_point - incident_point
        collision_momentum = collision_vector / np.linalg.norm(collision_vector) * scalar_translational_momentum
        Stationary(incident_molecule)
        incident_momenta = incident_molecule.get_momenta()
        incident_molecule.set_momenta(incident_momenta + collision_momentum)
        # move the mass center of incident molecule to the incident_point
        incident_molecule.set_positions(incident_molecule.get_positions() + incident_point)

        # combine target molecule and incident molecule
        self.current_step = 0
        self.current_system = target_molecule + incident_molecule
        write_xyz_file('MoREST.str', self.current_system)

        return self.current_step, self.current_system
            
    def get_current_structure(self):
        if self.sampling_parameters['sampling_initialization']:
            system = read_xyz_file(self.sampling_parameters['sampling_molecule'])
        else:
            system = self.current_traj[-1]
            #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            
        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses
        
        if self.morest_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_FD_forces(system, \
                                                      self.morest_parameters['fd_displacement'])
        else:
            self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return self.current_step, system
    

class scattering_velocity_Verlet(initialize_scattering):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE MD) dynamics.
    '''
    
    def __init__(self, morest_parameters, sampling_parameters, md_parameters, calculator=None, v_rescaling=False, sv_rescaling=False):
        super(velocity_Verlet, self).__init__(morest_parameters, sampling_parameters, calculator)
        self.md_parameters = md_parameters
        self.v_rescaling = v_rescaling
        self.sv_rescaling = sv_rescaling
        
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
        if self.morest_parameters['many_body_potential'].upper() in ['ML_FD'.upper()]:
            next_potential_energy, next_forces = self.many_body_potential.get_potential_FD_forces(next_system, \
                                                      self.morest_parameters['fd_displacement'])
        else:
            next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(next_system)
        
        ### v(t+dt) = v(t+0.5dt) + 0.5 * F(t+dt) * dt / m
        #next_accelerations = self.current_forces / self.masses
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        #next_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        next_system.set_momenta(next_momenta)
        
        #next_velocities = next_system.get_velocities()
        
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
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            self.current_traj.append(self.current_system)
            write_xyz_traj('MoREST_traj.xyz', self.current_system)
            kinetic_energy = self.current_system.get_kinetic_energy()
            if self.sv_rescaling:
                #self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses, self.K_simulation, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, self.Wt+R_t)
                self.d_Ee, self.Wt = write_SVR_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses, self.K_simulation, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            else:
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

def write_MD_log(MD_log, step, Ep, Ek, masses):
    n_atom = len(masses)
    #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
    #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
    T = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    Et = Ek + Ep
    MD_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')