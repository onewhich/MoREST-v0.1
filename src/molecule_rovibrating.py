import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_traj
from initialize_calculator import initialize_calculator
from numerical_integration import MD_integration
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
from kinetic_energy_assignment import get_kinetic_velocities, get_kinetic_temperatures, rescale_kinetic_temperature
from ase import units

class initialize_rovibrating(initialize_calculator):
    def __init__(self, morest_parameters, rovibrating_parameters, calculator=None, log_morest=None):

        self.log_file_name = 'MoREST_rovibrating.log'
        self.traj_file_name = 'MoREST_rovibrating_traj.xyz'

        super(initialize_rovibrating, self).__init__(morest_parameters, calculator, log_morest)
        self.rovibrating_parameters = rovibrating_parameters

        if self.rovibrating_parameters['rovibrating_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.set_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure()
        else:
            try:
                self.current_traj = read_xyz_traj(self.traj_file_name)
                self.current_step = (len(self.current_traj) - 1) * self.rovibrating_parameters['rovibrating_traj_interval']
                try:
                    self.ml_calculator.set_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.current_step = 0
                try:
                    self.ml_calculator.set_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure()
                
        self.integration = MD_integration(self.many_body_potential)
        
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(self.current_system)

        if self.current_step == 0:
            if not self.rovibrating_parameters['rovibrating_pre_thermalized']:
                if 'rovibrating_vibration_E' in self.rovibrating_parameters:
                    E_vibration = self.rovibrating_parameters['rovibrating_vibration_E']
                    T_vibration = 2/3 * E_vibration / units.kB / self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                elif 'rovibrating_vibration_T' in self.rovibrating_parameters:
                    E_vibration = 3/2 * units.kB * self.rovibrating_parameters['rovibrating_vibration_T'] * self.n_atom
                    T_vibration = self.rovibrating_parameters['rovibrating_vibration_T']
                if 'rovibrating_rotation_E' in self.rovibrating_parameters:
                    E_rotation = self.rovibrating_parameters['rovibrating_rotation_E']
                    T_rotation = 2/3 * E_rotation / units.kB / self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                elif 'rovibrating_rotation_T' in self.rovibrating_parameters:
                    E_rotation = 3/2 * units.kB * self.rovibrating_parameters['rovibrating_rotation_T'] * self.n_atom
                    T_rotation = self.rovibrating_parameters['rovibrating_rotation_T']
                E_kinetic = E_vibration + E_rotation
                T_thermalized = 2/3 * E_kinetic / units.kB / self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
                MaxwellBoltzmannDistribution(self.current_system, temperature_K = T_thermalized, force_temp = True)
                
                # There is a bug when getting translation and rotation velocities. The translation and rotation velocities are not orthogonal to each other and the vibration velocities.
                # But if the vibration energy is not lower too much than the rotation energy, the translation velocities can be cleaned,
                # and generate correct rotation and vibration velocities after several loops of the following process.
                for _ in range(9):
                    current_V_translation, current_V_rotation, current_V_vibration = get_kinetic_velocities(self.current_system)
                    current_T_translation, current_T_rotation, current_T_vibration = get_kinetic_temperatures(self.current_system)
                    new_V_translation = rescale_kinetic_temperature(current_V_translation, current_T_translation, 0)
                    new_V_rotation = rescale_kinetic_temperature(current_V_rotation, current_T_rotation, T_rotation)
                    new_V_vibration = rescale_kinetic_temperature(current_V_vibration, current_T_vibration, T_vibration)
                    self.current_system.set_velocities(new_V_translation + new_V_rotation + new_V_vibration)
                
            write_xyz_traj(self.traj_file_name, self.current_system)

        if self.rovibrating_parameters['rovibrating_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_system)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            
    def get_current_structure(self):
        if self.rovibrating_parameters['rovibrating_initialization']:
            system = read_xyz_file(self.rovibrating_parameters['rovibrating_molecule'])
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
                system = read_xyz_file(self.rovibrating_parameters['rovibrating_molecule'])

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        #self.current_accelerations = self.current_forces / self.masses

        #self.masses = system.get_masses()
        #self.current_accelerations = np.array([self.current_forces[i_atom] / self.masses[i_atom] for i_atom in range(self.n_atom)])
        
        return system
    
    def write_MD_log(self, log_file, step, system):
        Ep = system.get_potential_energy()
        Ek = system.get_kinetic_energy()
        T = system.get_temperature()
        Et = system.get_total_energy()
        log_file.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
        

class rovibrating_velocity_Verlet(initialize_rovibrating):
    '''
    This class implements velocity Verlet algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, rovibrating_parameters, calculator=None, log_morest=None):
        super(rovibrating_velocity_Verlet, self).__init__(morest_parameters, rovibrating_parameters, calculator, log_morest)
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.rovibrating_parameters['rovibrating_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces

        next_potential_energy, next_forces  = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.set_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.rovibrating_parameters['rovibrating_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.write_MD_log(self.MD_log, self.current_step, self.current_system)
        
        return self.current_step, self.current_system

class rovibrating_Suzuki_Yoshida_4th(initialize_rovibrating):
    '''
    This class implements Suzuki-Yoshida 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, rovibrating_parameters, calculator=None, log_morest=None):
        super(rovibrating_Suzuki_Yoshida_4th, self).__init__(morest_parameters, rovibrating_parameters, calculator, log_morest)
        
    def generate_new_step(self, bias_forces=None):
        time_step = self.rovibrating_parameters['rovibrating_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces

        next_potential_energy, next_forces  = self.integration.Suzuki_Yoshida_4th(time_step, self.current_system, self.current_forces, self.masses)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.set_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.rovibrating_parameters['rovibrating_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.write_MD_log(self.MD_log, self.current_step, self.current_system)
        
        return self.current_step, self.current_system

class rovibrating_Runge_Kutta_4th(initialize_rovibrating):
    '''
    This class implements Runge-Kutta 4th order algorithm to do microcanonical ensemble (NVE) dynamics.
    '''
    
    def __init__(self, morest_parameters, rovibrating_parameters, calculator=None, log_morest=None):
        super(rovibrating_Runge_Kutta_4th, self).__init__(morest_parameters, rovibrating_parameters, calculator, log_morest)
        
    def generate_new_step(self, bias_forces=None):
        '''
        This version comes from classic Runge-Kutta methods:
        Runge–Kutta methods. (2022, September 6). In Wikipedia. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
         https://www.haroldserrano.com/blog/visualizing-the-runge-kutta-method
        '''
        time_step = self.rovibrating_parameters['rovibrating_time_step']
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces
        
        next_potential_energy, next_forces  = self.integration.Runge_Kutta_4th(time_step, self.current_system, self.current_forces, self.masses)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
            
        try:
            self.ml_calculator.set_current_step(self.current_step)
        except:
            pass
        
        if self.current_step % self.rovibrating_parameters['rovibrating_traj_interval'] == 0:
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.write_MD_log(self.MD_log, self.current_step, self.current_system)
        
        return self.current_step, self.current_system
 