#from time import time
import numpy as np
#import sys
#sys.path.append('..')
from structure_io import  write_xyz_traj
from ase import units
from phase_space_sampling import MD
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, Langevin_velocity_rescaling, stochastic_velocity_rescaling
from barostat import barostat_space, Berendsen_volume_rescaling, Berendsen_enthalpy, Langevin_stage_1_propagate_thermostat, \
                    SVR_stage_1_propagate_thermostat, SVR_stage_2_propagate_momenta_eta, SVR_stage_3_propagate_position_volume, \
                    SVR_effective_enthalpy

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

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
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
                                            self.Nf, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                            self.Nf, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
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

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.Nf, self.T_simulation, \
                                                      self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        
        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.Nf, self.T_simulation, \
                                                      self.sampling_parameters['nvt_berendsen_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
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
        
        new_velocities, self.d_Ee = Langevin_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                self.Nf, self.sampling_parameters['nvt_Langevin_gamma'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, 0, self.d_Ee)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities, self.d_Ee = Langevin_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                self.Nf, self.sampling_parameters['nvt_Langevin_gamma'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
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
        
        new_velocities, self.d_Ee = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  self.Nf, self.sampling_parameters['nvt_svr_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            #self.d_Ee, self.Wt = self.write_MD_SVR_log_old(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), \
            #                                           self.masses, self.K_simulation, self.time_step, self.sampling_parameters['nvt_svr_tau'], 0, 0)
            self.Ee = self.write_MD_SVR_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, 0, self.d_Ee)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)
            self.Ee = 0
            #self.d_Ee = 0
            #self.Wt =  0

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)
            
        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)
        new_velocities, self.d_Ee = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  self.Nf, self.sampling_parameters['nvt_svr_tau'], self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
            
        self.update_step(next_potential_energy, next_forces)
        
        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
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

        super().__init__(morest_parameters, sampling_parameters, MD_parameters, molecule, self.traj_file_name, calculator, log_morest)

        self.NPH_space = barostat_space(self.MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_P = self.sampling_parameters['nph_svr_tau']
        self.eta = np.zeros(self.MD_parameters['barostat_number'])
        self.volume = np.zeros(self.MD_parameters['barostat_number'])
        self.P_current = np.zeros(self.MD_parameters['barostat_number'])
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.W_barostat = np.sum(self.masses) * self.tau_P**2
        if not self.sampling_parameters['sampling_initialization']:
            self.MD_parameters['barostat_space_size'] = self.current_system.info.get('barostat_space_size')

        Ek_atoms = self.NPH_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        self.lattice_vectors = self.current_system.get_cell()
        self.index_thermostat_atom = np.arange(self.n_atom, dtype=int)
        for i in range(self.MD_parameters['barostat_number']):
            coordinates_all = self.current_system.get_positions()
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            self.index_thermostat_atom = [atom for atom in self.index_thermostat_atom if atom not in index_atom]
            internal_virial = self.NPH_space.get_internal_virial(coordinates_all[index_atom], self.current_forces[index_atom], self.masses[index_atom])
            self.volume[i] = self.NPH_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPH_space.get_pressure(Ek_atoms[index_atom], internal_virial, self.volume[i])
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)')
            for i in range(self.MD_parameters['barostat_number']):
                self.MD_log.write(', Pressure (bar), Barostat size (A), Enthalpy (eV)')
            self.MD_log.write('\n')
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_simulation, self.MD_parameters['barostat_space_size'], H_effective)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        NPH_bias_forces = self.NPH_space.get_barostat_space_bias_forces()
        if (bias_forces == None) and (wall_potential != None):
            bias_forces = NPH_bias_forces + wall_potential(self.current_system.get_positions())
        elif (bias_forces != None) and (wall_potential == None):
            bias_forces += NPH_bias_forces
        elif (bias_forces != None) and (wall_potential != None):
            bias_forces += wall_potential(self.current_system.get_positions())
            bias_forces += NPH_bias_forces
        else:
            bias_forces = NPH_bias_forces
        total_forces = next_forces + bias_forces

        time_step = self.update_pre_step(time_step, None, None, updated_current_system)
        total_forces = self.fix_atoms(total_forces)
        half_time_step = 0.5 * time_step

        momenta_all = self.current_system.get_momenta()
        coordinates_all = self.current_system.get_positions()

        Ek_atoms = self.NPH_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        coordinates_all[self.index_thermostat_atom] = self.integration.propagate_positions_p_half(time_step, coordinates_all[self.index_thermostat_atom], \
                                                                                                 momenta_all[self.index_thermostat_atom], \
                                                                                                 self.masses[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = self.current_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]

            # stage 2: propagate 1/2 time step momenta & eta
            internal_virial = self.NPH_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPH_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPH_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)

            # stage 3: propagate 1 time step position, volume, momenta
            index_coordinates, current_volume, index_momenta, barostat_space_size = SVR_stage_3_propagate_position_volume(
                                                time_step, index_coordinates, index_momenta, current_eta, index_masses, current_volume, \
                                                self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], \
                                                self.NPH_space.barostat_space_center, self.MD_parameters['barostat_space_type'][i])
            self.MD_parameters['barostat_space_size'][i] = barostat_space_size

            self.eta[i] = current_eta
            coordinates_all[index_atom] = index_coordinates
            momenta_all[index_atom] = index_momenta

        # update forces
        self.current_system.set_positions(coordinates_all)
        self.current_system.set_momenta(momenta_all)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        total_forces = next_forces + bias_forces
        next_forces = self.fix_atoms(next_forces)
        total_forces = self.fix_atoms(total_forces)
        Ek_atoms = self.NPH_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)

        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = next_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]

            # stage 4: propagate 1/2 time step momenta & eta (again)
            internal_virial = self.NPH_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPH_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPH_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)

            self.eta[i] = current_eta
            self.volume[i] = current_volume
            momenta_all[index_atom] = index_momenta

        self.current_system.set_momenta(momenta_all)
        self.current_system.info['barostat_space_size'] = self.MD_parameters['barostat_space_size']
        
        self.NPH_space.update_barostat_space_wall()
        self.update_step(next_potential_energy, next_forces)
        
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_current, self.MD_parameters['barostat_space_size'], H_effective)
            
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

        self.NPT_space = barostat_space(self.MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_Berendsen_tau_t']
        self.tau_P = self.sampling_parameters['npt_Berendsen_tau_p']
        self.lattice_vectors = self.current_system.get_cell()
        # self.factor_Z = self.sampling_parameters['npt_Berendsen_compressibility']
        # factor_z (compressibility) and tau_p can be combined into single parameter, tau_P,
        # because factor_Z is only used in conjunction with tau_P.
        # init_miu = np.ones(self.MD_parameters['barostat_number']) # the first rescaling factor should be one for each barostat space
        if not self.sampling_parameters['sampling_initialization']:
            self.MD_parameters['barostat_space_size'] = self.current_system.info.get('barostat_space_size')

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.Nf, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        new_coordinates, P_current, volume = Berendsen_volume_rescaling(self.MD_parameters, self.time_step, self.current_system.get_positions(), \
                                                            self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, \
                                                            self.lattice_vectors)
        self.current_system.set_positions(new_coordinates)
        self.NPT_space.update_barostat_space_wall()

        H_enthalpy = Berendsen_enthalpy(self.current_system.get_kinetic_energy(), self.current_potential_energy, P_current, volume)

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)')
            for i in range(self.MD_parameters['barostat_number']):
                self.MD_log.write(', Pressure (bar), Barostat size (A), Enthalpy (eV)')
            self.MD_log.write('\n')
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
                                                self.MD_parameters['barostat_number'], P_current, self.MD_parameters['barostat_space_size'], H_enthalpy)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if bias_forces != None:
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces

        time_step = self.update_pre_step(time_step, bias_forces, wall_potential, updated_current_system)

        next_potential_energy, next_forces = self.integration.velocity_Verlet(time_step, self.current_system, self.current_forces, self.masses)

        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.Nf, \
                                                      self.T_simulation, self.tau_T, self.current_system.get_velocities())
        self.current_system.set_velocities(new_velocities)
        new_coordinates, P_current, volume = Berendsen_volume_rescaling(self.MD_parameters, self.time_step, self.current_system.get_positions(), \
                                                            self.current_system.get_forces(), new_velocities, self.masses, self.P_simulation, self.tau_P, \
                                                            self.lattice_vectors)
        self.current_system.set_positions(new_coordinates)
        self.current_system.info['barostat_space_size'] = self.MD_parameters['barostat_space_size']

        self.NPT_space.update_barostat_space_wall()
        self.update_step(next_potential_energy, next_forces)

        H_enthalpy = Berendsen_enthalpy(self.current_system.get_kinetic_energy(), self.current_potential_energy, P_current, volume)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
                                                self.MD_parameters['barostat_number'], P_current, self.MD_parameters['barostat_space_size'], H_enthalpy)
        
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

        self.NPT_space = barostat_space(self.MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_Langevin_tau_t']
        self.tau_P = self.sampling_parameters['npt_Langevin_tau_p']
        self.eta = np.zeros(self.MD_parameters['barostat_number'])
        self.volume = np.zeros(self.MD_parameters['barostat_number'])
        self.P_current = np.zeros(self.MD_parameters['barostat_number'])
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf_all = 3*self.n_atom - 2
        self.W_barostat = self.Nf_all * units.kB * self.T_simulation * self.tau_P**2
        if not self.sampling_parameters['sampling_initialization']:
            self.MD_parameters['barostat_space_size'] = self.current_system.info.get('barostat_space_size')

        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        self.lattice_vectors = self.current_system.get_cell()
        self.index_thermostat_atom = np.arange(self.n_atom, dtype=int)
        for i in range(self.MD_parameters['barostat_number']):
            coordinates_all = self.current_system.get_positions()
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            self.index_thermostat_atom = [atom for atom in self.index_thermostat_atom if atom not in index_atom]
            internal_virial = self.NPT_space.get_internal_virial(coordinates_all[index_atom], self.current_forces[index_atom], self.masses[index_atom])
            self.volume[i] = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, self.volume[i])
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        self.Nf_thermostat_atom = self.correction_degree_of_freedom(self.index_thermostat_atom, 3*len(self.index_thermostat_atom))

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)')
            for i in range(self.MD_parameters['barostat_number']):
                self.MD_log.write(', Pressure (bar), Barostat size (A), Enthalpy (eV)')
            self.MD_log.write('\n')
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_simulation, self.MD_parameters['barostat_space_size'], H_effective)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if (bias_forces == None) and (wall_potential != None):
            bias_forces = NPT_bias_forces + wall_potential(self.current_system.get_positions())
        elif (bias_forces != None) and (wall_potential == None):
            bias_forces += NPT_bias_forces
        elif (bias_forces != None) and (wall_potential != None):
            bias_forces += wall_potential(self.current_system.get_positions())
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces
        total_forces = self.current_forces + bias_forces

        time_step = self.update_pre_step(time_step, None, None, updated_current_system)
        total_forces = self.fix_atoms(total_forces)
        half_time_step = 0.5 * time_step

        momenta_all = self.current_system.get_momenta()
        coordinates_all = self.current_system.get_positions()

        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        coordinates_all[self.index_thermostat_atom] = self.integration.propagate_positions_p_half(time_step, coordinates_all[self.index_thermostat_atom], \
                                                                                                 momenta_all[self.index_thermostat_atom], \
                                                                                                 self.masses[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = self.current_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]
            index_Nf = self.correction_degree_of_freedom(index_atom, 3*len(index_atom)-2)

            # stage 1: propagate 1/2 time step thermostat
            index_momenta, current_eta = Langevin_stage_1_propagate_thermostat(half_time_step, index_masses, self.T_simulation, \
                                                     index_Nf, self.tau_T, index_momenta, current_eta, self.W_barostat)

            # stage 2: propagate 1/2 time step momenta & eta
            internal_virial = self.NPT_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)

            # stage 3: propagate 1 time step position, volume, momenta
            index_coordinates, current_volume, index_momenta, barostat_space_size = SVR_stage_3_propagate_position_volume(
                                                time_step, index_coordinates, index_momenta, current_eta, index_masses, current_volume, \
                                                self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], \
                                                self.NPT_space.barostat_space_center, self.MD_parameters['barostat_space_type'][i])
            self.MD_parameters['barostat_space_size'][i] = barostat_space_size

            self.eta[i] = current_eta
            coordinates_all[index_atom] = index_coordinates
            momenta_all[index_atom] = index_momenta

        # update forces
        self.current_system.set_positions(coordinates_all)
        self.current_system.set_momenta(momenta_all)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        total_forces = next_forces + bias_forces
        next_forces = self.fix_atoms(next_forces)
        total_forces = self.fix_atoms(total_forces)
        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)

        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = next_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]
            index_Nf = self.correction_degree_of_freedom(index_atom, 3*len(index_atom)-2)

            # stage 4: propagate 1/2 time step momenta & eta (again)
            internal_virial = self.NPT_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)
            
            # stage 5: propagate 1/2 time step thermostat (again)
            index_momenta, current_eta = Langevin_stage_1_propagate_thermostat(half_time_step, index_masses, self.T_simulation, \
                                                     index_Nf, self.tau_T, index_momenta, current_eta, self.W_barostat)

            self.eta[i] = current_eta
            self.volume[i] = current_volume
            momenta_all[index_atom] = index_momenta

        new_velocities, self.d_Ee = Langevin_velocity_rescaling(time_step, np.sum(Ek_atoms[self.index_thermostat_atom]), \
                                                                  self.K_simulation * (3*len(self.index_thermostat_atom)) / self.Nf, \
                                                                  3*len(self.index_thermostat_atom), self.tau_T, \
                                                                  self.current_system.get_velocities()[self.index_thermostat_atom])
        momenta_all[self.index_thermostat_atom] = new_velocities * self.masses[self.index_thermostat_atom]

        self.current_system.set_momenta(momenta_all)
        self.current_system.info['barostat_space_size'] = self.MD_parameters['barostat_space_size']

        self.NPT_space.update_barostat_space_wall()
        self.update_step(next_potential_energy, next_forces)
        
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_current, self.MD_parameters['barostat_space_size'], H_effective)
            
        return self.current_step, self.current_system


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

        self.NPT_space = barostat_space(self.MD_parameters, self.current_system)

        self.P_simulation = self.MD_parameters['barostat_pressure']
        self.tau_T = self.sampling_parameters['npt_svr_tau_t']
        self.tau_P = self.sampling_parameters['npt_svr_tau_p']
        self.eta = np.zeros(self.MD_parameters['barostat_number'])
        self.volume = np.zeros(self.MD_parameters['barostat_number'])
        self.P_current = np.zeros(self.MD_parameters['barostat_number'])
        # N_f = 3 * N - 3 + 1, remove the center of mass DOF, add the barostat volume DOF
        self.Nf_all = 3*self.n_atom - 2
        self.W_barostat = self.Nf_all * units.kB * self.T_simulation * self.tau_P**2
        if not self.sampling_parameters['sampling_initialization']:
            self.MD_parameters['barostat_space_size'] = self.current_system.info.get('barostat_space_size')

        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        self.lattice_vectors = self.current_system.get_cell()
        self.index_thermostat_atom = np.arange(self.n_atom, dtype=int)
        for i in range(self.MD_parameters['barostat_number']):
            coordinates_all = self.current_system.get_positions()
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            self.index_thermostat_atom = [atom for atom in self.index_thermostat_atom if atom not in index_atom]
            internal_virial = self.NPT_space.get_internal_virial(coordinates_all[index_atom], self.current_forces[index_atom], self.masses[index_atom])
            self.volume[i] = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, self.volume[i])
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        self.Nf_thermostat_atom = self.correction_degree_of_freedom(self.index_thermostat_atom, 3*len(self.index_thermostat_atom))

        if self.sampling_parameters['sampling_initialization']:
            self.MD_log = open(self.log_file_name, 'w', buffering=1)
            self.MD_log.write('# MD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)')
            for i in range(self.MD_parameters['barostat_number']):
                self.MD_log.write(', Pressure (bar), Barostat size (A), Enthalpy (eV)')
            self.MD_log.write('\n')
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.current_system.get_kinetic_energy(), self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_current, self.MD_parameters['barostat_space_size'], H_effective)
        else:
            self.MD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, wall_potential=None, updated_current_system=None):
        NPT_bias_forces = self.NPT_space.get_barostat_space_bias_forces()
        if (bias_forces == None) and (wall_potential != None):
            bias_forces = NPT_bias_forces + wall_potential(self.current_system.get_positions())
        elif (bias_forces != None) and (wall_potential == None):
            bias_forces += NPT_bias_forces
        elif (bias_forces != None) and (wall_potential != None):
            bias_forces += wall_potential(self.current_system.get_positions())
            bias_forces += NPT_bias_forces
        else:
            bias_forces = NPT_bias_forces
        total_forces = self.current_forces + bias_forces

        time_step = self.update_pre_step(time_step, None, None, updated_current_system)
        total_forces = self.fix_atoms(total_forces)
        half_time_step = 0.5 * time_step

        momenta_all = self.current_system.get_momenta()
        coordinates_all = self.current_system.get_positions()

        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)
        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        coordinates_all[self.index_thermostat_atom] = self.integration.propagate_positions_p_half(time_step, coordinates_all[self.index_thermostat_atom], \
                                                                                                 momenta_all[self.index_thermostat_atom], \
                                                                                                 self.masses[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = self.current_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]
            index_Nf = self.correction_degree_of_freedom(index_atom, 3*len(index_atom)-2)

            # stage 1: propagate 1/2 time step thermostat
            alpha = SVR_stage_1_propagate_thermostat(half_time_step, np.sum(Ek_atoms[index_atom]), self.T_simulation, \
                                                     index_Nf, self.tau_T, current_eta, self.W_barostat)
            index_momenta *= alpha
            current_eta *= alpha

            # stage 2: propagate 1/2 time step momenta & eta
            internal_virial = self.NPT_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)

            # stage 3: propagate 1 time step position, volume, momenta
            index_coordinates, current_volume, index_momenta, barostat_space_size = SVR_stage_3_propagate_position_volume(
                                                time_step, index_coordinates, index_momenta, current_eta, index_masses, current_volume, \
                                                self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], \
                                                self.NPT_space.barostat_space_center, self.MD_parameters['barostat_space_type'][i])
            self.MD_parameters['barostat_space_size'][i] = barostat_space_size

            self.eta[i] = current_eta
            coordinates_all[index_atom] = index_coordinates
            momenta_all[index_atom] = index_momenta

        # update forces
        self.current_system.set_positions(coordinates_all)
        self.current_system.set_momenta(momenta_all)
        next_potential_energy, next_forces = self.many_body_potential.get_potential_forces(self.current_system)
        total_forces = next_forces + bias_forces
        next_forces = self.fix_atoms(next_forces)
        total_forces = self.fix_atoms(total_forces)
        Ek_atoms = self.NPT_space.get_atom_kinetic_energies(self.current_system.get_velocities(), self.masses)

        momenta_all[self.index_thermostat_atom] = self.integration.propagate_momenta_half(time_step, momenta_all[self.index_thermostat_atom], \
                                                                                         total_forces[self.index_thermostat_atom])
        for i in range(self.MD_parameters['barostat_number']):
            index_atom = self.MD_parameters['barostat_action_atoms'][i]
            current_eta = self.eta[i]
            index_momenta = momenta_all[index_atom]
            index_coordinates = coordinates_all[index_atom]
            index_forces = next_forces[index_atom]
            index_total_forces = total_forces[index_atom]
            index_masses = self.masses[index_atom]
            index_Nf = self.correction_degree_of_freedom(index_atom, 3*len(index_atom)-2)

            # stage 4: propagate 1/2 time step momenta & eta (again)
            internal_virial = self.NPT_space.get_internal_virial(index_coordinates, index_forces, index_masses)
            current_volume = self.NPT_space.get_volume(self.MD_parameters['barostat_space_shape'][i], self.MD_parameters['barostat_space_size'][i], self.lattice_vectors)
            self.P_current[i] = self.NPT_space.get_pressure(Ek_atoms[index_atom], internal_virial, current_volume)
            T_current = self.current_system.get_temperature()
            current_eta, index_momenta = SVR_stage_2_propagate_momenta_eta(half_time_step, current_eta, index_momenta, current_volume, self.P_current[i], \
                                                            self.P_simulation[i], T_current, self.W_barostat, index_forces, index_total_forces, index_masses)
            
            # stage 5: propagate 1/2 time step thermostat (again)
            alpha = SVR_stage_1_propagate_thermostat(half_time_step, np.sum(Ek_atoms[index_atom]), self.T_simulation, \
                                                     index_Nf, self.tau_T, current_eta, self.W_barostat)
            index_momenta *= alpha
            current_eta *= alpha

            self.eta[i] = current_eta
            self.volume[i] = current_volume
            momenta_all[index_atom] = index_momenta

        new_velocities, self.d_Ee = stochastic_velocity_rescaling(time_step, np.sum(Ek_atoms[self.index_thermostat_atom]), \
                                                                  self.K_simulation * self.Nf_thermostat_atom / self.Nf, \
                                                                  self.Nf_thermostat_atom, self.tau_T, \
                                                                  self.current_system.get_velocities()[self.index_thermostat_atom])
        momenta_all[self.index_thermostat_atom] = new_velocities * self.masses[self.index_thermostat_atom]

        self.current_system.set_momenta(momenta_all)
        self.current_system.info['barostat_space_size'] = self.MD_parameters['barostat_space_size']

        self.NPT_space.update_barostat_space_wall()
        self.update_step(next_potential_energy, next_forces)
        
        Ek_t = self.current_system.get_kinetic_energy()
        Ep_t = self.current_potential_energy
        H_effective = SVR_effective_enthalpy(Ek_t, Ep_t, self.eta, self.volume, self.T_simulation, self.P_current, self.W_barostat)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_NPT_log(self.MD_log, self.current_step, self.current_potential_energy, self.kinetic_energy, self.masses, \
                                                self.MD_parameters['barostat_number'], self.P_current, self.MD_parameters['barostat_space_size'], H_effective)

        return self.current_step, self.current_system
