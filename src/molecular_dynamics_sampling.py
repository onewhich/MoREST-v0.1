#from time import time
import numpy as np
#import sys
#sys.path.append('..')
from structure_io import  write_xyz_traj
from ase import units
from phase_space_sampling import MD
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, stochastic_velocity_rescaling
from barostat import barostat_space, Berendsen_volume_rescaling, stochastic_velocity_volume_rescaling

class NVE_VV(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system

class NVK_VR(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NVT_Berendsen(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        
        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NVT_Langevin(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  1, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), self.d_Ee, R_t)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NVT_SVR(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NPH_SVR(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        self.NPH_space = barostat_space(MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_P = self.sampling_parameters['nph_svr_tau']
        self.eta = np.zeros(self.MD_parameters['barostat_number']) # initialize the velocity of the barostat
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf = 3*self.n_atom - 2
        self.half_time_step = self.time_step/2
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        self.W_barostat = self.Nf * units.kB * T_current * self.tau_P**2

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
        
        NPT_bias_forces = self.NPH_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)

        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        # stage 2: propagate 1/2 time step velocities
        # stage 3: propagate time step positions and velocities
        # stage 4: propagate 1/2 time step velocities
        new_coordinates, new_momenta, self.eta, P_current = stochastic_velocity_volume_rescaling(self.MD_parameters, self.time_step, self.half_time_step, \
                                                            self.current_system.get_positions(), self.current_system.get_forces(), self.current_system.get_velocities(), \
                                                            self.eta, self.current_system.get_momenta(), self.masses, self.W_barostat, T_current, self.P_simulation)
        self.current_system.set_positions(new_coordinates)
        self.current_system.set_momenta(new_momenta)
        self.NPH_space.update_barostat_space_wall()
        
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system


class NPT_Berendsen(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        self.NPT_space = barostat_space(MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_Berendsen_tau_t']
        self.tau_P = self.sampling_parameters['npt_Berendsen_tau_p']
        self.beta = self.sampling_parameters['npt_Berendsen_compressibility']
        init_miu = np.ones(self.MD_parameters['barostat_number']) # the first rescaling factor should be one for each barostat space

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        next_coordinates, self.miu, P_current = Berendsen_volume_rescaling(self.MD_parameters, self.time_step, self.current_system.get_positions(), \
                                                               self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, self.beta, init_miu)
        self.current_system.set_positions(next_coordinates)
        self.NPT_space.update_barostat_space_wall()

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)

        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        next_coordinates, self.miu, P_current = Berendsen_volume_rescaling(self.MD_parameters, self.time_step, self.current_system.get_positions(), \
                                                               self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, self.beta, self.miu)
        self.current_system.set_positions(next_coordinates)
        self.NPT_space.update_barostat_space_wall()
        
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses)
        
        return self.current_step, self.current_system


class NPT_Langevin(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        self.NPT_space = barostat_space(MD_parameters, self.current_system)


class NPT_SVR(MD):
    def __init__(self, morest_parameters, sampling_parameters, MD_parameters, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_MD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_MD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, T_simulation, calculator, log_morest)

        self.NPT_space = barostat_space(MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_svr_tau_t']
        self.tau_P = self.sampling_parameters['npt_svr_tau_p']
        self.eta = np.zeros(self.MD_parameters['barostat_number']) # initialize the velocity of the barostat
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf = 3*self.n_atom - 2
        self.half_time_step = self.time_step/2
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        self.W_barostat = self.Nf * units.kB * T_current * self.tau_P**2

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, updated_current_system)
        
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if type(bias_forces) != type(None):
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)

        # stage 1: propagate 1/2 time step thermostat
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step/2, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  self.Nf, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        self.eta *= alpha
        
        # stage 2: propagate 1/2 time step velocities
        # stage 3: propagate time step positions and velocities
        # stage 4: propagate 1/2 time step velocities
        T_current = self.get_temperature(self.current_system.get_kinetic_energy(), self.n_atom)
        new_coordinates, new_momenta, self.eta, P_current = stochastic_velocity_volume_rescaling(self.MD_parameters, self.time_step, self.half_time_step, \
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
        
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            #print(next_coordinates) #DEGUB
            #print(next_forces)    #DEBUG
            #self.current_traj.append(self.current_system)
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
            #                                           self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], self.d_Ee, R_t)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system
