import numpy as np
from structure_io import write_xyz_file, write_xyz_traj
from phase_space_sampling import RPMD_normal_mode
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, Langevin_velocity_rescaling, stochastic_velocity_rescaling
from ase import units

class RP_NVE_normal_mode(RPMD_normal_mode):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')
            self.write_MD_log(self.RPMD_log, self.current_step, np.mean(self.current_beads_potential_energy), \
                              np.mean(self.get_beads_kinetic_energy(self.current_beads)), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        time_step = self.RPMD_update_pre_step(time_step, bias_forces, updated_current_beads)

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, \
                                                self.C_jk, self.n_beads, self.omega_k, self.atom_masses)#, NVT_SVR=False)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.write_MD_log(self.RPMD_log, self.current_step, np.mean(self.current_beads_potential_energy), \
                              np.mean(self.get_beads_kinetic_energy(self.current_beads)), self.masses)

        return self.current_step, self.current_system
    
class RP_NVT_SVR_normal_mode(RPMD_normal_mode):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.mean(self.current_beads_potential_energy), \
                                            np.mean(self.get_beads_kinetic_energy(self.current_beads)), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        time_step = self.RPMD_update_pre_step(time_step, bias_forces, updated_current_beads)

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, \
                                                self.C_jk, self.n_beads, self.omega_k, self.atom_masses)
                                                    #NVT_SVR=True, thermostat_SVR=self.thermostat_SVR)

        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        tmp_d_Ee_list = []
        for i in range(self.n_beads):
            old_velocities = self.current_beads[i].get_velocities()
            new_velocities, tmp_d_Ee = stochastic_velocity_rescaling(self.time_step, self.current_beads[i].get_kinetic_energy(), self.K_simulation, \
                                                                    3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], old_velocities)
            self.current_beads[i].set_velocities(new_velocities)
            tmp_d_Ee_list.append(tmp_d_Ee)
        self.d_Ee = np.mean(tmp_d_Ee_list)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.mean(self.current_beads_potential_energy), \
                                            np.mean(self.get_beads_kinetic_energy(self.current_beads)), self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system
    
    def thermostat_SVR(self, current_momenta_k):
        '''
        Applies Stochastic Velocity Rescaling (SVR) to internal modes (k > 0).
        Does not work well, discarded.
        '''
        for k in range(1, self.n_beads):  # skip centroid (k = 0)
            for i in range(self.n_atom):
                p = current_momenta_k[k, i]
                m = self.atom_masses[i]
                KE = 0.5 * np.sum(p**2) / m
                dof = 3  # degrees of freedom per atom

                if KE == 0:
                    continue  # avoid division by zero

                c = np.exp(-self.time_step / self.sampling_parameters['nvt_svr_tau'])
                T_inst = (2 * KE) / (dof * units.kB)
                xi = np.random.normal()
                chi = np.random.chisquare(dof - 1)
                alpha_sq = c + (1 - c) * (self.T_simulation / T_inst) * (chi + xi**2) / (dof)
                alpha = np.sqrt(alpha_sq)

                current_momenta_k[k, i] = alpha * p
        return current_momenta_k