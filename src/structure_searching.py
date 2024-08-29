import numpy as np
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_traj, write_xyz_file
from initialize_calculator import initialize_calculator
from ase.md.velocitydistribution import Stationary, ZeroRotation
from ase import units

class initialize_searching(initialize_calculator):
    def __init__(self, morest_parameters, searching_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(initialize_searching, self).__init__(morest_parameters, calculator, log_morest)
        self.searching_parameters = searching_parameters
        self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_searching_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        if self.searching_parameters['searching_initialization']:
            self.current_step = 0
            try:
                self.ml_calculator.get_current_step(self.current_step)
            except:
                pass
            self.current_system = self.get_current_structure(molecule)
            self.potential_energy_list = []
            self.potential_energy_list.append(self.current_system.get_potential_energy())
            write_xyz_traj(self.traj_file_name, self.current_system)
        else:
            try:
                self.current_traj = read_xyz_traj(self.traj_file_name)
                self.current_step = len(self.current_traj) - 1
                try:
                    self.ml_calculator.get_current_step(self.current_step)
                except:
                    pass
                self.current_system = self.get_current_structure()
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
                write_xyz_traj(self.traj_file_name, self.current_system)
            
    def get_current_structure(self, molecule=None):
        if self.searching_parameters['searching_initialization']:
            if molecule == None:
                system = read_xyz_file(self.searching_parameters['searching_starting_point'])
            else:
                system = molecule
        else:
            try:
                system = self.current_traj[-1]
            except:
                self.log_morest.write('Can not read current structure, and read structure from starting point.')
                if molecule == None:
                    system = read_xyz_file(self.searching_parameters['searching_starting_point'])
                else:
                    system = molecule

        self.n_atom = system.get_global_number_of_atoms()
        self.masses = system.get_masses()[:,np.newaxis]
        self.kinetic_energy = system.get_kinetic_energy()
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(system)
        self.current_convergence = np.max(np.linalg.norm(self.current_forces,axis=-1))
        
        return system
    
    def check_divergence(self):
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

class gradient_descent(initialize_searching):
    '''
    This class implements steepest descent algorithm for structure optimization.
    '''
    def __init__(self, morest_parameters, searching_parameters, gradient_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, \
                 method=None, log_morest=None):
        super().__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        self.method = method.upper()
        self.step_size = gradient_parameters['gradient_step_size']
        self.p_k = self.current_forces
        if self.method == 'BFGS':
            #self.H_k = np.array([np.identity(3) for i in range(self.n_atom)])
            self.I = np.eye(3*self.n_atom, dtype=int)
            self.H_k = self.I
            #self.p_k = np.dot(self.H_k, self.current_forces.flatten()).reshape(np.shape(self.current_forces))

        if self.searching_parameters['searching_initialization']:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_'+self.method+'.log', 'w', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'w', buffering=1)
            self.searching_log.write('# MD step, Potential energy (eV), dE (eV), MAX atomic force (eV/A)\n')   
            self.write_log()
        else:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_'+self.method+'.log', 'a', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        if updated_current_system != None:
            self.current_system = updated_current_system
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces

        current_coordinates = self.current_system.get_positions()

        # r(k+1) = r(k) + a * p(k) 
        next_coordinates = current_coordinates + self.step_size * self.p_k
        self.current_system.set_positions(next_coordinates)

        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)

        if bias_forces != None:
            next_forces = next_forces + bias_forces

        if self.method == 'BFGS':
            # s(k) = r(k+1) - r(k)
            s_k = (next_coordinates - current_coordinates).flatten()
            # y(k) = F(k) - F(k+1)
            y_k = (self.current_forces - next_forces).flatten()
            # rho(k) = 1/(y(k)^T @ s(k))
            rho_k = 1/(y_k @ s_k)
            # H(k+1) = (I - rho(k) s(k) y(k).T) H(k) (I - rho(k) y(k) s(k).T) + rho(k) s(k) s(k).T
            self.H_k = (self.I - rho_k * np.outer(s_k, y_k)) @ self.H_k @ (self.I - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)
            # p(k+1) = F(k+1) + H(k+1) @ F(k+1)
            self.p_k = next_forces + (self.H_k @ next_forces.flatten()).reshape(np.shape(next_forces))
            ### self.p_k = next_forces + np.dot(self.H_k, next_gradient).reshape(np.shape(next_forces))
        elif self.method == 'CG':
            # beta(k+1) = (F(k+1).T @ (F(k+1)-F(k))) / (F(k).T @ F(k))
            next_beta = [(next_forces[i] @ (next_forces[i]-self.current_forces[i])) / (self.current_forces[i] @ self.current_forces[i]) \
                         for i in range(self.n_atom)]
            next_beta = np.array(next_beta)[:,np.newaxis]
            # p(k+1) = F(k+1) + beta(k+1) * p(k)
            self.p_k = next_forces + next_beta * self.p_k
        elif self.method == 'GD':
            self.p_k = next_forces

        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces,axis=-1))
        self.potential_energy_list.append(self.current_potential_energy)

        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point']+'_new', self.current_system)
        self.write_log()

        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system
    
    def write_log(self):
        Ep = self.potential_energy_list[-1]
        if len(self.potential_energy_list) < 2:
            dE = 0.
        else:
            dE = self.potential_energy_list[-1] - self.potential_energy_list[-2]
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
            if len(dE) >= 1:
                dE = dE[0]
        except:
            pass
        self.searching_log.write(str(self.current_step)+'    '+str(Ep)+'    '+str(dE)+'    '+str(self.current_convergence)+'\n')

    
class searching_velocity_Verlet(initialize_searching):
    '''
    This class implements velocity Verlet algorithm for structure optimization methods.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST_searching.xyz (default name) records the initial xyz structure of the system
    MoREST_searching.xyz_new (default name) records the current xyz structure of the system
    '''
    def __init__(self, morest_parameters, searching_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(searching_velocity_Verlet, self).__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)

    def VV_next_step(self, bias_forces=None, updated_current_system=None):
        if type(self.time_step) != float:
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
        self.potential_energy_list.append(self.current_system.get_potential_energy())
        self.kinetic_energy = self.current_system.get_kinetic_energy()

        Stationary(self.current_system)
        #ZeroRotation(self.current_system)
            
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point']+'_new', self.current_system)
        

class FIRE_velocity_Verlet(searching_velocity_Verlet):
    '''
    This class implements FIRE structure optimization algorithm based on velocity Verlet integrator.
    '''
    def __init__(self, morest_parameters, searching_parameters, FIRE_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(FIRE_velocity_Verlet, self).__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        self.FIRE_parameters = FIRE_parameters
        if self.FIRE_parameters['fire_equal_masses']:
            masses = np.ones(self.n_atom)
            self.current_system.set_masses(masses)
            self.masses = masses[:,np.newaxis]
        self.time_step = self.FIRE_parameters['fire_time_step'] * np.ones(self.n_atom)
        self.max_time_step = self.FIRE_parameters['fire_max_time_step']
        self.alpha = self.FIRE_parameters['fire_alpha_init'] * np.ones(self.n_atom)
        self.N_min = self.FIRE_parameters['fire_N_min']
        self.f_increase = self.FIRE_parameters['fire_f_increase']
        self.f_decrease = self.FIRE_parameters['fire_f_decrease']
        self.f_alpha = self.FIRE_parameters['fire_f_alpha']
        self.N_negative =  np.zeros(self.n_atom, dtype=int)

        if self.searching_parameters['searching_initialization']:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_FIRE.log', 'w', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'w', buffering=1)
            self.searching_log.write('# MD step, Potential energy (eV),  dE (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), MAX atomic force (eV/A)\n')   
            self.write_FIRE_log()
        else:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_FIRE.log', 'a', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'a', buffering=1)
                
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
                self.alpha[i] = self.FIRE_parameters['fire_alpha_init']

            next_velocities.append(i_next_velocity)
            #print(self.current_step, P, self.N_negative[i], self.time_step[i] / units.fs, self.alpha[i], i_next_velocity)
        self.current_system.set_velocities(np.array(next_velocities))

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        self.VV_next_step(bias_forces, updated_current_system)
        self.FIRE()
        self.write_FIRE_log()
        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system

    def write_FIRE_log(self):
        Ep = self.current_potential_energy
        if len(self.potential_energy_list) < 2:
            dE = 0.
        else:
            dE = self.potential_energy_list[-1] - self.potential_energy_list[-2]
        Ek = self.kinetic_energy
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
            if len(dE) >= 1:
                dE = dE[0]
        except:
            pass
        #Ek = np.sum([0.5 * masses[i] * np.linalg.norm(velocities[i])**2 for i in range(n_atom)])
        #Ek = np.sum(0.5 * masses * np.linalg.norm(velocities)**2)
        T = 2/3 * Ek/units.kB /self.n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        Et = Ek + Ep
        self.searching_log.write(str(self.current_step)+'    '+ \
                                  str(Ep)+'    '+str(dE)+'    '+ \
                                   str(Ek)+'    '+str(T)+'    '+ \
            str(Et)+'    '+str(self.current_convergence)+'    '+'\n')
        #for i in range(self.n_atom):
        #    self.searching_log.write(str(self.time_step[i]/units.fs)+'    ')
        #self.searching_log.write('\n')
        
