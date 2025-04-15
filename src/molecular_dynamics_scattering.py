from structure_io import write_xyz_traj
from trajectory_scattering import initialize_scattering
from numerical_integraion import MD_integration
from kinetic_energy_assignment import clean_translation

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
            next_momenta[0:self.n_atom_target] = clean_translation(next_momenta[0:self.n_atom_target])
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
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
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
            next_momenta[0:self.n_atom_target] = clean_translation(next_momenta[0:self.n_atom_target])
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
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
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
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses)
        
        return self.current_step, self.current_system
