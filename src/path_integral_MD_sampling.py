import numpy as np
from structure_io import write_xyz_file, write_xyz_traj
from phase_space_sampling import RPMD
from thermostat import velocity_rescaling, Berendsen_velocity_rescaling, stochastic_velocity_rescaling
from barostat import barostat_space, Berendsen_volume_rescaling, stochastic_velocity_volume_rescaling

class RP_NVE(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, self.masses)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVK_VR(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, self.masses)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = velocity_rescaling(self.sampling_parameters['nvk_vr_dt'], self.T_simulation, self.current_system.get_kinetic_energy(), \
                                        self.n_atom, old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVT_Berendsen(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')   
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, self.masses)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities = Berendsen_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.n_atom, \
                                                      self.T_simulation, self.sampling_parameters['nvt_berendsen_tau'], old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)
        
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.write_MD_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses)

        return self.current_step, self.current_system
    
class RP_NVT_Langevin(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, self.masses)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()

        # only rescale the centroids velocities
        old_velocities = self.current_system.get_velocities()
        new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
                                                                  1, 1/(2*self.sampling_parameters['nvt_Langevin_gamma']), old_velocities)
        self.current_system.set_velocities(new_velocities)
        d_velocities = new_velocities - old_velocities
        for i in range(self.n_beads):
            tmp_velocites = self.current_beads[i].get_velocities()
            self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system
    
class RP_NVT_SVR(RPMD):
    def __init__(self, morest_parameters, sampling_parameters, RPMD_parameters, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
                
        if type(log_file_name) == type(None):
            self.log_file_name = 'MoREST_RPMD.log'
        else:
            self.log_file_name = log_file_name
        if type(traj_file_name) == type(None):
            self.traj_file_name = 'MoREST_RPMD_traj.xyz'
        else:
            self.traj_file_name = traj_file_name

        self.RPMD_clean_translation = RPMD_parameters['rpmd_clean_translation']
        self.RPMD_clean_rotation = RPMD_parameters['rpmd_clean_rotation']

        super().__init__(morest_parameters, sampling_parameters, RPMD_parameters, molecule, traj_file_name, calculator, log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.RPMD_log = open(self.log_file_name, 'w', buffering=1)
            self.RPMD_log.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV), Effective energy (eV)\n')   
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.current_system.get_kinetic_energy(), self.masses)
        else:
            self.RPMD_log = open(self.log_file_name, 'a', buffering=1)

    def generate_new_step(self, time_step=None, bias_forces=None, updated_current_beads=None):
        if type(updated_current_beads) != type(None):
            self.current_beads = updated_current_beads
        
        ### F(t) + bias
        if type(bias_forces) != type(None):
            for i in range(self.n_beads):
                current_forces = self.current_beads_forces[i]
                self.current_beads_forces[i] = current_forces + bias_forces
            
        if type(time_step) == type(None):
            time_step = self.time_step

        self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta = \
            self.integration.RP_velocity_Verlet(time_step, self.current_beads, self.current_beads_forces, self.masses)
        
        self.RPMD_update_step(self.current_beads_potential_energy, self.current_beads_forces, current_beads_positions, current_beads_momenta)

        if self.RPMD_clean_rotation:
            self.clean_rotation_centroid()
        if self.RPMD_clean_translation:
            self.clean_translation_centroid()


        ## only rescale the centroids 
        #old_velocities = self.current_system.get_velocities()
        #new_velocities, self.d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_system.get_kinetic_energy(), self.K_simulation, \
        #                                                          3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], old_velocities)
        #self.current_system.set_velocities(new_velocities)
        #d_velocities = new_velocities - old_velocities
        #for i in range(self.n_beads):
        #    tmp_velocites = self.current_beads[i].get_velocities()
        #    self.current_beads[i].set_velocities(tmp_velocites + d_velocities)

        tmp_d_Ee_list = []
        for i in range(self.n_beads):
            old_velocities = self.current_beads[i].get_velocities()
            new_velocities, tmp_d_Ee, alpha = stochastic_velocity_rescaling(self.time_step, self.current_beads[i].get_kinetic_energy(), self.K_simulation, \
                                                                    3*self.n_atom, self.sampling_parameters['nvt_svr_tau'], old_velocities)
            self.current_beads[i].set_velocities(new_velocities)
            tmp_d_Ee_list.append(tmp_d_Ee)
        self.d_Ee = np.average(tmp_d_Ee_list)
            
        write_xyz_file(self.sampling_parameters['sampling_molecule']+'_new', self.current_system)

        if self.current_step % self.sampling_parameters['sampling_traj_interval'] == 0:
            for i in range(self.n_beads):
                write_xyz_traj(self.beads_file_head+"traj_"+str(i)+'.xyz',self.current_beads[i])
            write_xyz_traj(self.traj_file_name, self.current_system)
            self.kinetic_energy = self.current_system.get_kinetic_energy()
            self.Ee = self.write_MD_SVR_log(self.RPMD_log, self.current_step, np.average(self.current_beads_potential_energy), self.kinetic_energy, self.masses, self.Ee, self.d_Ee)
            
        return self.current_step, self.current_system