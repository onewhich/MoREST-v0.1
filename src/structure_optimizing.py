import numpy as np
from structure import read_xyz_file, read_xyz_traj, write_xyz_traj, write_xyz_file
from many_body_potential import ml_potential, on_the_fly, molpro_calculator
from ase.md.velocitydistribution import Stationary, ZeroRotation
from ase import units
#from copy import deepcopy

class initialize_optimizing:
    def __init__(self, morest_parameters, optimizing_parameters, calculator=None, log_morest=None):
        self.morest_parameters = morest_parameters
        self.optimizing_parameters = optimizing_parameters
        self.log_morest = log_morest
        
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
            self.ml_calculator = ml_potential(ab_initio_calculator = calculator, \
                                    ml_parameters = self.morest_parameters, \
                                    log_file = self.log_morest)
            self.many_body_potential = on_the_fly(self.ml_calculator)
            
        else:
            raise Exception('Which many body potential will you use?')
            
    def get_current_structure(self, molecule=None):
        if self.optimizing_parameters['optimizing_initialization']:
            if molecule == None:
                system = read_xyz_file(self.optimizing_parameters['optimizing_starting_point'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
                #system = read_xyz_file('MoREST.str_new') #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
            except:
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
                if molecule == None:
                    system = read_xyz_file(self.optimizing_parameters['optimizing_starting_point'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        self.current_convergence = np.max(np.linalg.norm(self.current_forces,axis=-1))
        
        return system
    

class optimizing_velocity_Verlet(initialize_optimizing):
    '''
    This class implements velocity Verlet algorithm for structure optimizing methods.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST.str (default name) records the initial xyz structure of the system
    MoREST.str_new (default name) records the current xyz structure of the system
    '''
    def __init__(self, morest_parameters, optimizing_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(fire_velocity_Verlet, self).__init__(morest_parameters, optimizing_parameters, calculator, log_morest)
        self.optimizing_parameters = optimizing_parameters
        self.traj_file_name = traj_file_name
        self.log_file_name = log_file_name
        self.log_morest = log_morest
        if self.optimizing_parameters['optimizing_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
            self.potential_energy_list = []
            self.potential_energy_list.append(self.current_system.get_potential_energy())
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
                self.current_step = len(self.current_traj) - 1
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
                self.potential_energy_list = [i_sys.get_potential_energy() for i_sys in self.current_traj]
                self.potential_energy_list.append(self.current_system.get_potential_energy())
            except:
                self.current_step = 0
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure(molecule)
                self.potential_energy_list = []
                self.potential_energy_list.append(self.current_system.get_potential_energy())
                if self.traj_file_name == None:
                    write_xyz_traj('MoREST_traj.xyz', self.current_system)
                else:
                    write_xyz_traj(self.traj_file_name, self.current_system)

    def VV_next_step(self, bias_forces=None, updated_current_system=None):
        time_step = self.time_step[:,np.newaxis]
        
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
        #next_velocities = current_velocities + 0.5 * (self.current_accelerations + next_accelerations) * time_step
        #self.current_system.set_velocities(next_velocities)
        
        ### p(t+dt) = p(t+0.5dt) + 0.5 * F(t+dt) * dt
        next_momenta = momenta_half + 0.5 * next_forces * time_step
        self.current_system.set_momenta(next_momenta)
        
        #next_velocities = next_system.get_velocities()
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces,axis=-1))

        Stationary(self.current_system)
        #ZeroRotation(self.current_system)
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        self.potential_energy_list.append(self.current_system.get_potential_energy())
        write_xyz_traj('MoREST_traj.xyz', self.current_system)
        write_xyz_file(self.optimizing_parameters['optimizing_starting_point']+'_new', self.current_system)
        kinetic_energy = self.current_system.get_kinetic_energy()
        write_optimizing_log(self.optimizing_log, self.current_step, self.current_potential_energy, kinetic_energy, self.masses,self.current_convergence)
        
        if self.potential_energy_list[-1] > self.potential_energy_list[0]:
            if self.potential_energy_list[-2] > self.potential_energy_list[-1]:
                if self.potential_energy_list[-3] > self.potential_energy_list[-2]:
                    if self.potential_energy_list[-4] > self.potential_energy_list[-3]:
                        if self.potential_energy_list[-5] > self.potential_energy_list[-4]:
                            try:
                                self.log_morest.write('The optimization has an abnormal energy rise. The mission of MoREST is terminated.\n')
                            except:
                                pass
                            raise Exception('The optimization has an abnormal energy rise. The mission is terminated.')
        

class fire_velocity_Verlet(optimizing_velocity_Verlet):
    '''
    This class implements FIRE structure optimization algorithm based on velocity Verlet integrator.
    '''
    def __init__(self, morest_parameters, optimizing_parameters, fire_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(fire_velocity_Verlet, self).__init__(morest_parameters, optimizing_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        self.optimizing_parameters = optimizing_parameters
        self.fire_parameters = fire_parameters
        self.traj_file_name = traj_file_name
        self.log_file_name = log_file_name
        self.log_morest = log_morest
        if self.optimizing_parameters['optimizing_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
            self.potential_energy_list = []
            self.potential_energy_list.append(self.current_system.get_potential_energy())
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
                self.current_step = len(self.current_traj) - 1
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure() #TODO: need to read current step and system from MoREST.str_new instead of MoREST_traj.xyz
                self.potential_energy_list = [i_sys.get_potential_energy() for i_sys in self.current_traj]
                self.potential_energy_list.append(self.current_system.get_potential_energy())
            except:
                self.current_step = 0
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure(molecule)
                self.potential_energy_list = []
                self.potential_energy_list.append(self.current_system.get_potential_energy())
                if self.traj_file_name == None:
                    write_xyz_traj('MoREST_traj.xyz', self.current_system)
                else:
                    write_xyz_traj(self.traj_file_name, self.current_system)

        if self.optimizing_parameters['optimizing_initialization']:
            if self.log_file_name == None:
                self.optimizing_log = open('MoREST_FIRE.log', 'w', buffering=1)
            else:
                self.optimizing_log = open(self.log_file_name, 'w', buffering=1)
            self.optimizing_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), MAX atomic force (eV/A)\n')   
            write_optimizing_log(self.optimizing_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, self.current_convergence)
        else:
            if self.log_file_name == None:
                self.optimizing_log = open('MoREST_FIRE.log', 'a', buffering=1)
            else:
                self.optimizing_log = open(self.log_file_name, 'a', buffering=1)


        if self.fire_parameters['fire_equal_masses']:
            masses = np.ones(self.n_atom)
            self.current_system.set_masses(masses)
            self.masses = masses[:,np.newaxis]
        self.time_step = self.fire_parameters['fire_time_step'] * np.ones(self.n_atom)
        self.max_time_step = self.fire_parameters['fire_max_time_step']
        self.alpha = self.fire_parameters['fire_alpha_init'] * np.ones(self.n_atom)
        self.N_min = self.fire_parameters['fire_N_min']
        self.f_increase = self.fire_parameters['fire_f_increase']
        self.f_decrease = self.fire_parameters['fire_f_decrease']
        self.f_alpha = self.fire_parameters['fire_f_alpha']
        self.N_negative =  np.zeros(self.n_atom, dtype=int)
                
    def FIRE(self):
        '''
        This version comes from paper:
        @article{bitzek2006structural,
          title={Structural relaxation made simple},
          author={Bitzek, Erik and Koskinen, Pekka and G{\"a}hler, Franz and Moseler, Michael and Gumbsch, Peter},
          journal={Physical review letters},
          volume={97},
          number={17},
          pages={170201},
          year={2006},
          publisher={APS}
        }
        '''
        current_velocities = self.current_system.get_velocities()
        next_velocities = []
        for i in range(self.n_atom):
            i_force = self.current_forces[i]
            i_velocity = current_velocities[i]

            # F1: P = F \dot v
            P = np.dot(i_force, i_velocity)

            # F2: v = (1-alpha)*v + alpha * F * |v|
            i_next_velocity = (1-self.alpha[i])*i_velocity + self.alpha[i]*i_force*np.linalg.norm(i_velocity)

            # F3: if P > 0
            if P > 0 and self.N_negative[i] > self.N_min:
                self.time_step[i] = min(self.time_step[i]*self.f_increase, self.max_time_step)
                self.alpha[i] *= self.f_alpha

            # F4: if P < 0
            elif P <= 0:
                self.N_negative[i] += 1
                self.time_step[i] *= self.f_decrease
                i_next_velocity *= 0
                self.alpha[i] = self.fire_parameters['fire_alpha_init']

            next_velocities.append(i_next_velocity)
            #print(self.current_step, P, self.N_negative[i], self.time_step[i] / units.fs, self.alpha[i], i_next_velocity)
        self.current_system.set_velocities(np.array(next_velocities))

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.FIRE()

        return self.current_convergence, self.current_step, self.current_system

def write_optimizing_log(optimizing_log, step, Ep, Ek, masses, convergence):
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
    optimizing_log.write(str(step)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'    '+str(convergence)+'\n')