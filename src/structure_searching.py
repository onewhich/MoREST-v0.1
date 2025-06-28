import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from structure_io import read_xyz_file, read_xyz_traj, write_xyz_traj, write_xyz_file
from initialize_calculator import initialize_calculator
from numerical_integraion import MD_integration
# Stationary and ZeroRotation from ase will not change the total kinetic energy, the vibrational energy will arise after these two processes.
from kinetic_energy_assignment import clean_translation
from ase import units

class initialize_searching(initialize_calculator):
    def __init__(self, morest_parameters, searching_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(initialize_searching, self).__init__(morest_parameters, calculator, log_morest)
        self.searching_parameters = searching_parameters
        self.log_file_name = log_file_name
        if traj_file_name == None:
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
        elif self.method == 'SGD':
            self.sgd_fraction = gradient_parameters['sgd_fraction']
        elif self.method == 'ADAM':
            self.t = 0
            self.beta1 = gradient_parameters['adam_beta1']
            self.beta2 = gradient_parameters['adam_beta2']
            self.eps = gradient_parameters['adam_eps']
            self.m = np.zeros(3*self.n_atom)
            self.v = np.zeros(3*self.n_atom)

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
            denominator = y_k @ s_k
            if abs(denominator) > 1e-10:
                # rho(k) = 1/(y(k)^T @ s(k))
                rho_k = 1/denominator
                # H(k+1) = (I - rho(k) s(k) y(k).T) H(k) (I - rho(k) y(k) s(k).T) + rho(k) s(k) s(k).T
                self.H_k = (self.I - rho_k * np.outer(s_k, y_k)) @ self.H_k @ (self.I - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)
                #### p(k+1) = F(k+1) + H(k+1) @ F(k+1)
                #### self.p_k = next_forces + (self.H_k @ next_forces.flatten()).reshape(np.shape(next_forces))
                self.p_k = - (self.H_k @ next_forces.flatten()).reshape(np.shape(next_forces))
        elif self.method == 'CG':
            beta_list = []
            for i in range(self.n_atom):
                denominator = self.current_forces[i] @ self.current_forces[i]
                if denominator < 1e-10:
                    beta = 0
                else:
                    # beta(k+1) = (F(k+1).T @ (F(k+1)-F(k))) / (F(k).T @ F(k))
                    beta = (next_forces[i] @ (next_forces[i] - self.current_forces[i])) / denominator
                beta_list.append(beta)
            next_beta = np.array(beta_list)[:, np.newaxis]
            # p(k+1) = F(k+1) + beta(k+1) * p(k)
            self.p_k = next_forces + next_beta * self.p_k
        elif self.method == 'GD':
            self.p_k = next_forces
        elif self.method == 'SGD':
            # 随机选一部分原子或坐标维度进行更新
            total_coords = self.n_atom * 3
            num_selected = max(1, int(self.sgd_fraction * total_coords))
            flat_forces = next_forces.flatten()
            selected_indices = np.random.choice(total_coords, size=num_selected, replace=False)
            # 创建一个新的 p_k 向量，只在选中的维度使用 force，其他设为 0
            sgd_p_k = np.zeros_like(flat_forces)
            sgd_p_k[selected_indices] = flat_forces[selected_indices]
            self.p_k = sgd_p_k.reshape((self.n_atom, 3))
        elif self.method == 'ADAM':
            self.t += 1
            g = next_forces.flatten()  # 梯度
            # 更新一阶矩和二阶矩估计
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)
            # 偏差修正
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            # 计算更新步长
            step = - self.step_size * m_hat / (np.sqrt(v_hat) + self.eps)
            self.p_k = step.reshape((self.n_atom, 3))

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

class L_BFGS_descent(gradient_descent):
    '''
    Implements the L-BFGS (Limited-memory BFGS) structure optimization algorithm.
    '''
    def __init__(self, morest_parameters, searching_parameters, gradient_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, \
                 method='L-BFGS', log_morest=None):
        super().__init__(morest_parameters, searching_parameters, gradient_parameters, molecule, log_file_name, traj_file_name, calculator, \
                         method, log_morest)

        self.m = gradient_parameters['lbfgs_history_step']  # Number of historical steps to store
        self.s_list = []  # List of position differences s_k
        self.y_list = []  # List of gradient differences y_k
        self.rho_list = []  # List of 1 / (y_k^T * s_k)

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        if updated_current_system != None:
            self.current_system = updated_current_system

        if bias_forces != None:
            self.current_forces += bias_forces

        current_coordinates = self.current_system.get_positions()
        current_gradient = -self.current_forces.flatten()  # Negative gradient for descent direction

        # Two-loop recursion to compute the L-BFGS direction
        q = current_gradient.copy()
        alpha_list = []
        for s_k, y_k, rho_k in reversed(list(zip(self.s_list, self.y_list, self.rho_list))):
            alpha_k = rho_k * np.dot(s_k, q)
            q -= alpha_k * y_k
            alpha_list.append(alpha_k)

        if self.y_list:
            y_last = self.y_list[-1]
            s_last = self.s_list[-1]
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        else:
            gamma = 1.0

        r = gamma * q
        for i in range(len(self.s_list)):
            beta = self.rho_list[i] * np.dot(self.y_list[i], r)
            r += self.s_list[i] * (alpha_list[-(i+1)] - beta)

        p_k = r.reshape(self.current_forces.shape)
        next_coordinates = current_coordinates + self.step_size * p_k
        self.current_system.set_positions(next_coordinates)

        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)

        s_k = (next_coordinates - current_coordinates).flatten()
        y_k = (-next_forces.flatten() - current_gradient)
        rho_k_denom = np.dot(y_k, s_k)

        if abs(rho_k_denom) > 1e-10:
            rho_k = 1.0 / rho_k_denom
            self.s_list.append(s_k)
            self.y_list.append(y_k)
            self.rho_list.append(rho_k)

            # Keep only the most recent `m` entries
            if len(self.s_list) > self.m:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)

        self.p_k = -next_forces
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces, axis=-1))
        self.potential_energy_list.append(self.current_potential_energy)

        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point'] + '_new', self.current_system)
        self.write_log()
        
        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system


class scipy_L_BFGS_B_descent(initialize_searching):
    """
    Structure optimization using SciPy's L-BFGS-B algorithm.
    This class is compatible with the MoREST optimization framework.
    """

    def __init__(self, morest_parameters, searching_parameters, gradient_parameters,
                 molecule=None, log_file_name=None, traj_file_name=None,
                 calculator=None, log_morest=None):
        super().__init__(morest_parameters, searching_parameters, molecule,
                         log_file_name, traj_file_name, calculator, log_morest)

        self.max_iter = gradient_parameters['searching_max_steps']
        self.tol = gradient_parameters['searching_convergence']

        if self.log_file_name == None:
            self.searching_log = open('MoREST_scipy_L-BFGS-B.log', 'w', buffering=1)
        else:
            self.searching_log = open(self.log_file_name, 'w', buffering=1)

        self.searching_log.write('# SciPy L-BFGS-B optimization log\n')
        self.searching_log.write('# MD step, Potential energy (eV), MAX atomic force (eV/A)\n')

    def _energy_and_grad(self, flat_coords):
        """
        Compute energy and gradient for SciPy minimizer from flat coordinates.
        """
        positions = flat_coords.reshape((-1, 3))
        self.current_system.set_positions(positions)
        energy, forces = self.many_body_potential.get_potential_forces(self.current_system)

        self.current_forces = forces
        self.current_potential_energy = energy
        self.current_convergence = np.max(np.linalg.norm(forces, axis=-1))
        self.potential_energy_list.append(energy)

        self.write_log()
        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point'] + '_new', self.current_system)

        return energy, -forces.flatten()

    def optimize(self):
        """
        Run the full L-BFGS-B optimization using SciPy.
        """
        initial_positions = self.current_system.get_positions().flatten()

        result = minimize(self._energy_and_grad,
                          initial_positions,
                          method='L-BFGS-B',
                          jac=True,
                          tol=self.tol,
                          options={'disp': True, 'maxiter': self.max_iter})

        # Update system with final coordinates
        final_positions = result.x.reshape((-1, 3))
        self.current_system.set_positions(final_positions)
        self.current_forces = -result.jac.reshape((-1, 3))
        self.current_potential_energy = result.fun
        self.current_convergence = np.max(np.linalg.norm(self.current_forces, axis=-1))

        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point'] + '_final', self.current_system)
        self.write_log()

    def write_log(self):
        Ep = self.current_potential_energy
        Fmax = self.current_convergence
        self.searching_log.write(f'{self.current_step}    {Ep:.6f}    {Fmax:.6f}\n')
        self.current_step += 1


class searching_velocity_Verlet(initialize_searching):
    '''
    This class implements velocity Verlet algorithm for structure optimization methods.
    MoREST_traj.xyz records the trajectory in an extended xyz format
    MoREST_searching.xyz (default name) records the initial xyz structure of the system
    MoREST_searching.xyz_new (default name) records the current xyz structure of the system
    '''
    def __init__(self, morest_parameters, searching_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(searching_velocity_Verlet, self).__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        self.integration = MD_integration(self.many_body_potential)

    def VV_next_step(self, bias_forces=None, updated_current_system=None):
        if type(self.time_step) != float:
            time_step = self.time_step[:,np.newaxis]
        
        if updated_current_system != None:
            self.current_system = updated_current_system
        
        ### F(t) + bias
        if bias_forces != None:
            self.current_forces = self.current_forces + bias_forces
        
        next_potential_energy, next_forces  = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces,axis=-1))
        self.potential_energy_list.append(self.current_system.get_potential_energy())
        self.kinetic_energy = self.current_system.get_kinetic_energy()

        #clean_rotation(self.current_system)
        clean_translation(self.current_system)
            
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
        

class BFGS_TS(gradient_descent):
    """
    Transition State search using full BFGS with min-mode correction (BFGS-TS).
    """
    def __init__(self, morest_parameters, searching_parameters, gradient_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, \
                 method='BFGS-TS', log_morest=None):
        super().__init__(morest_parameters, searching_parameters, gradient_parameters, molecule, log_file_name, traj_file_name, calculator, \
                         method, log_morest)
        self.I = np.eye(3 * self.n_atom)
        self.H_k = self.I.copy()

    def min_mode_direction(self, hessian):
        # Compute the eigenvector corresponding to the most negative eigenvalue
        eigval, eigvec = eigsh(hessian, k=1, which='SA')  # Smallest algebraic
        return eigvec.flatten()

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        if updated_current_system is not None:
            self.current_system = updated_current_system

        if bias_forces is not None:
            self.current_forces += bias_forces

        current_coords = self.current_system.get_positions().flatten()

        # Estimate approximate Hessian using current H_k
        hessian = self.H_k.copy()

        # Get the minimum-mode direction
        min_mode = self.min_mode_direction(hessian)

        # Project forces and flip along min-mode direction
        force_flat = self.current_forces.flatten()
        proj = np.dot(force_flat, min_mode)
        force_flat -= 2 * proj * min_mode  # Reverse the component along min-mode

        # BFGS update
        step = self.step_size * force_flat
        next_coords = current_coords + step
        self.current_system.set_positions(next_coords.reshape((-1, 3)))

        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        if bias_forces is not None:
            next_forces += bias_forces

        s_k = step
        y_k = self.current_forces.flatten() - next_forces.flatten()
        rho = 1.0 / (np.dot(y_k, s_k) + 1e-12)  # avoid div by 0
        I = self.I
        self.H_k = (I - rho * np.outer(s_k, y_k)) @ self.H_k @ (I - rho * np.outer(y_k, s_k)) + rho * np.outer(s_k, s_k)

        self.p_k = -force_flat.reshape((-1, 3))
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces, axis=-1))
        self.potential_energy_list.append(self.current_potential_energy)

        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point'] + '_new', self.current_system)
        self.write_log()

        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system


class L_BFGS_TS(gradient_descent):
    """
    Limited-memory BFGS for Transition State search (L-BFGS-TS).
    """
    def __init__(self, morest_parameters, searching_parameters, gradient_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, \
                 method='L-BFGS-TS', log_morest=None):
        super().__init__(morest_parameters, searching_parameters, gradient_parameters, molecule, log_file_name, traj_file_name, calculator, \
                         method, log_morest)
        self.history_s = []
        self.history_y = []
        self.m = gradient_parameters['lbfgs_history_step']  # memory size

    def min_mode_direction(self, system):
        # Use finite difference approximation of Hessian
        coords = system.get_positions().flatten()
        epsilon = 1e-3
        grad0 = self.current_forces.flatten()
        perturb = np.random.randn(len(coords))
        system.set_positions((coords + epsilon * perturb).reshape((-1, 3)))
        _, grad1 = self.many_body_potential.get_potential_forces(system)
        grad1 = grad1.flatten()
        diff_grad = (grad1 - grad0) / epsilon
        return diff_grad / np.linalg.norm(diff_grad)

    def two_loop_recursion(self, q):
        alpha = []
        rho = []
        for s_k, y_k in reversed(list(zip(self.history_s, self.history_y))):
            rho_k = 1.0 / (np.dot(y_k, s_k) + 1e-10)
            rho.append(rho_k)
            a = rho_k * np.dot(s_k, q)
            alpha.append(a)
            q -= a * y_k

        # Initial H_0 = Identity scaled
        if self.history_y:
            y_last = self.history_y[-1]
            s_last = self.history_s[-1]
            H0 = np.dot(y_last, s_last) / (np.dot(y_last, y_last) + 1e-10)
        else:
            H0 = 1.0
        r = H0 * q

        for s_k, y_k, rho_k, a_k in zip(self.history_s, self.history_y, rho[::-1], alpha[::-1]):
            b = rho_k * np.dot(y_k, r)
            r += s_k * (a_k - b)
        return r

    def generate_new_step(self, bias_forces=None, updated_current_system=None):
        if updated_current_system is not None:
            self.current_system = updated_current_system

        if bias_forces is not None:
            self.current_forces += bias_forces

        coords = self.current_system.get_positions().flatten()
        force_flat = self.current_forces.flatten()

        # Estimate min-mode direction
        min_mode = self.min_mode_direction(self.current_system)
        proj = np.dot(force_flat, min_mode)
        modified_grad = force_flat - 2 * proj * min_mode

        # L-BFGS search direction
        search_dir = -self.two_loop_recursion(modified_grad)

        step = self.step_size * search_dir
        new_coords = coords + step
        self.current_system.set_positions(new_coords.reshape((-1, 3)))
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        if bias_forces is not None:
            next_forces += bias_forces

        s_k = step
        y_k = force_flat - next_forces.flatten()
        if np.dot(s_k, y_k) > 1e-10:
            self.history_s.append(s_k)
            self.history_y.append(y_k)
            if len(self.history_s) > self.m:
                self.history_s.pop(0)
                self.history_y.pop(0)

        self.p_k = -search_dir.reshape((-1, 3))
        self.current_step += 1
        self.current_forces = next_forces
        self.current_potential_energy = next_potential_energy
        self.current_convergence = np.max(np.linalg.norm(self.current_forces, axis=-1))
        self.potential_energy_list.append(self.current_potential_energy)

        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass

        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point'] + '_new', self.current_system)
        self.write_log()
        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system


class dimer(initialize_searching):
    '''
    This class implements the Dimer method for transition state searching,
    driven by an L-BFGS optimizer. It is designed to find first-order saddle points.
    '''
    def __init__(self, morest_parameters, searching_parameters, dimer_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        """
        Initializes the Dimer_LBFGS optimizer.

        Args:
            morest_parameters: General parameters for MoREST.
            searching_parameters: Parameters for the searching process.
            dimer_parameters: A dictionary with specific parameters for the Dimer method.
                Expected keys:
                'dimer_distance' (float): The distance between the two dimer replicas. Default: 0.01 A.
                'dimer_rotation_step' (float): Step size for the rotational force. Default: 0.01.
            molecule: An ASE Atoms object for the initial structure.
            log_file_name (str): Name for the log file.
            traj_file_name (str): Name for the trajectory file.
            calculator: The calculator object.
            log_morest: The main MoREST log file handle.
        """
        super().__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        
        self.dimer_parameters = dimer_parameters
        self.dimer_distance = self.dimer_parameters['dimer_distance']
        self.rot_dt = self.dimer_parameters['dimer_rotation_step']

        # Initialize Dimer direction vector (randomly and normalized)
        # It's flattened to work with scipy's optimizer
        self.dimer_direction = np.random.rand(self.n_atom * 3)
        self.dimer_direction /= np.linalg.norm(self.dimer_direction)

        self.curvature = 0.0 # To store the curvature along the dimer direction
        
        # Setup logging
        if self.searching_parameters['searching_initialization']:
            if self.log_file_name is None:
                self.searching_log = open('MoREST_Dimer.log', 'w', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'w', buffering=1)
            self.searching_log.write('# Step, Potential Energy (eV), dE (eV), MAX Force (eV/A), Curvature (eV/A^2)\n')
            self.write_log()
        else:
            if self.log_file_name is None:
                self.searching_log = open('MoREST_Dimer.log', 'a', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'a', buffering=1)

    def _objective_function(self, flat_coords):
        """
        This is the core function passed to the L-BFGS optimizer.
        It returns the true potential energy and a MODIFIED gradient (force)
        that guides the optimizer towards the saddle point.
        """
        # 1. Update system with new coordinates from the optimizer
        self.current_system.set_positions(flat_coords.reshape(-1, 3))

        # 2. Perform the Dimer rotation to find the lowest curvature mode
        # Create two replica systems
        coords_mid_flat = self.current_system.get_positions().flatten()
        R1_coords = (coords_mid_flat - self.dimer_distance / 2 * self.dimer_direction).reshape(-1, 3)
        R2_coords = (coords_mid_flat + self.dimer_distance / 2 * self.dimer_direction).reshape(-1, 3)

        # Create temporary ASE Atoms objects for the replicas
        # This assumes your calculator can handle new Atoms objects
        R1_system = self.current_system.copy()
        R1_system.set_positions(R1_coords)
        R2_system = self.current_system.copy()
        R2_system.set_positions(R2_coords)
        
        # Calculate forces on the replicas (the most expensive step)
        _, F1 = self.many_body_potential.get_potential_forces(R1_system)
        _, F2 = self.many_body_potential.get_potential_forces(R2_system)
        F1_flat, F2_flat = F1.flatten(), F2.flatten()
        
        # Calculate the rotational force (torque) and rotate the dimer direction
        force_diff_parallel = np.dot(F1_flat - F2_flat, self.dimer_direction) * self.dimer_direction
        torque = (F1_flat - F2_flat) - force_diff_parallel
        self.dimer_direction += self.rot_dt * torque
        self.dimer_direction /= np.linalg.norm(self.dimer_direction)
        
        # 3. Calculate true energy and the modified force for the translation step
        self.current_potential_energy, self.current_forces = self.many_body_potential.get_potential_forces(self.current_system)
        
        # The modified force inverts the component along the dimer direction
        force_mid_flat = self.current_forces.flatten()
        force_parallel_comp = np.dot(force_mid_flat, self.dimer_direction)
        modified_force = force_mid_flat - 2 * force_parallel_comp * self.dimer_direction
        
        # 4. Update internal state for logging and return values to optimizer
        self.current_convergence = np.max(np.linalg.norm(self.current_forces, axis=-1))
        # Recalculate curvature with the final rotated direction for accuracy
        self.curvature = np.dot(F1_flat - F2_flat, self.dimer_direction) / self.dimer_distance

        return self.current_potential_energy, -modified_force # Scipy minimizes, so we provide -F_mod

    def _callback(self, current_coords_flat):
        """
        A callback function called by scipy.optimize.minimize after each major iteration.
        Used for logging, writing trajectories, and checking for divergence.
        """
        self.potential_energy_list.append(self.current_potential_energy)
        self.current_step += 1
        
        try:
            self.ml_calculator.get_current_step(self.current_step)
        except:
            pass
        
        # Write logs and trajectories
        self.write_log()
        write_xyz_traj(self.traj_file_name, self.current_system)
        write_xyz_file(self.searching_parameters['searching_starting_point']+'_new', self.current_system)
        
        self.check_divergence()
        
    def generate_new_step(self, fmax=0.01, max_steps=200):
        """
        This method initiates the entire L-BFGS optimization run for the Dimer method.
        Unlike other optimizers, this is not a single step but the full process.

        Args:
            fmax (float): The force convergence criterion (max force component).
            max_steps (int): The maximum number of optimization steps.
        
        Returns:
            The final convergence value, total steps, and the final system object.
        """
        initial_coords_flat = self.current_system.get_positions().flatten()
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_coords_flat,
            method='L-BFGS-B',
            jac=True,  # Our function returns the jacobian (gradient/force)
            callback=self._callback,
            options={
                'gtol': fmax, # Gradient tolerance
                'maxiter': max_steps
            }
        )
        
        if result.success:
            self.log_morest.write('Dimer optimization converged successfully.\n')
        else:
            self.log_morest.write(f'Dimer optimization finished without convergence: {result.message}\n')
            
        final_system = self.current_system.copy()
        final_system.set_positions(result.x.reshape(-1, 3))
        
        return self.current_convergence, self.current_step, final_system

    def write_log(self):
        """
        Writes a log entry for the Dimer optimization step.
        """
        Ep = self.potential_energy_list[-1]
        if len(self.potential_energy_list) < 2:
            dE = 0.
        else:
            dE = self.potential_energy_list[-1] - self.potential_energy_list[-2]
        
        try:
            if hasattr(Ep, "__len__") and len(Ep) >= 1:
                Ep = Ep[0]
            if hasattr(dE, "__len__") and len(dE) >= 1:
                dE = dE[0]
        except:
            pass
            
        self.searching_log.write(f"{self.current_step:<6d} {Ep:<22.8f} {dE:<12.8f} "
                                 f"{self.current_convergence:<18.8f} {self.curvature:<18.8f}\n")
        

class GAD_velocity_Verlet(searching_velocity_Verlet):
    '''
    Implements Gentle Ascent Dynamics (GAD) for transition state search.
    Based on the velocity Verlet integrator.
    '''
    def __init__(self, morest_parameters, searching_parameters, GAD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        super(GAD_velocity_Verlet, self).__init__(morest_parameters, searching_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        self.GAD_parameters = GAD_parameters
        self.time_step = self.GAD_parameters['gad_time_step'] * np.ones(self.n_atom)

        if self.searching_parameters['searching_initialization']:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_GAD.log', 'w', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'w', buffering=1)
            self.searching_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Max atomic force (eV/A)\n')
        else:
            if self.log_file_name == None:
                self.searching_log = open('MoREST_GAD.log', 'a', buffering=1)
            else:
                self.searching_log = open(self.log_file_name, 'a', buffering=1)

    def apply_GAD_forces(self):
        """
        Modify forces using Gentle Ascent Dynamics:
        Project out component along the maximal force direction and reverse it.
        """
        forces = self.current_forces.copy()
        norms = np.linalg.norm(forces, axis=1)
        max_force_idx = np.argmax(norms)
        max_force_vector = forces[max_force_idx]
        max_force_unit = max_force_vector / np.linalg.norm(max_force_vector)

        # Project all forces onto the max force unit direction
        projected = np.dot(forces, max_force_unit)[:, np.newaxis] * max_force_unit[np.newaxis, :]

        # Reverse the component along the maximum force
        gad_forces = forces - 2.0 * projected  # F_perp + (-F_parallel)
        return gad_forces

    def generate_new_step(self, updated_current_system=None):
        # Apply GAD modified forces
        gad_forces = self.apply_GAD_forces()

        # Proceed to next step with modified forces
        self.VV_next_step(bias_forces=gad_forces, updated_current_system=updated_current_system)
        self.write_GAD_log()

        self.check_divergence()

        return self.current_convergence, self.current_step, self.current_system

    def write_GAD_log(self):
        Ep = self.current_potential_energy
        Ek = self.kinetic_energy
        try:
            if len(Ep) >= 1:
                Ep = Ep[0]
        except:
            pass
        T = 2 / 3 * Ek / units.kB / self.n_atom
        Et = Ep + Ek
        self.searching_log.write(str(self.current_step) + '    ' +
                                 str(Ep) + '    ' +
                                 str(Ek) + '    ' +
                                 str(T) + '    ' +
                                 str(Et) + '    ' +
                                 str(self.current_convergence) + '\n')
