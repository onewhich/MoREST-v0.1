import numpy as np
#import scipy.constants
from json import JSONEncoder
from ase import units

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class read_parameters:
    '''
    The parameters reading module
    --------
    INPUT:
        parameter_file: parameter file name. Default name is 'MoREST.parameter' in current directory.
    --------
    Enhanced sampling parameters are stored in dictionary named 'enhanced_sampling_parameters'.
    ITS parameters are stored in dictionary named 'its_parameters'.
    Wall potential parameters are stored in dictionary named 'wall_potential_parameters'.
    '''
    
    def __init__(self, parameter_file='MoREST.in'):
        try:
            __parameters = open(parameter_file,'r').readlines()
        except FileNotFoundError:
            raise Exception('Can not find the parameter file: '+parameter_file)
        
        self.morest_parameters = {}
        self.morest_parameters['morest_initialization'] = False
        self.morest_parameters['morest_save_parameters_file'] = False
        self.morest_parameters['morest_load_parameters_file'] = False
        self.morest_parameters['ml_active_learning'] = False
        self.morest_parameters['ml_energy_uncertainty_tolerance'] = 0.01
        self.morest_parameters['ml_fd_forces'] = True
        self.morest_parameters['fd_displacement'] = 0.0025
        self.sampling_parameters = {}
        self.sampling_parameters['phase_space_sampling'] = False
        self.sampling_parameters['sampling_initialization'] = False
        self.sampling_parameters['sampling_molecule'] = 'MoREST.str'
        self.md_parameters = {}
        self.md_parameters['md_clean_translation'] = True
        self.md_parameters['md_clean_rotation'] = False
        self.scattering_parameters = {}
        self.scattering_parameters['trajectory_scattering'] = False
        self.scattering_parameters['scattering_initialization'] = False
        self.scattering_parameters['scattering_pre_thermolized'] = False
        self.scattering_parameters['scattering_traj_stop'] = None
        self.scattering_parameters['scattering_traj_length'] = None
        self.scattering_parameters['scattering_target_molecule'] = 'MoREST.str_target'
        self.scattering_parameters['scattering_incident_molecule'] = 'MoREST.str_incident'
        self.enhanced_sampling_parameters = {}
        self.enhanced_sampling_parameters['enhanced_sampling'] = False
        self.re_parameters = {}
        self.re_parameters['re_initialization'] = False
        self.re_parameters['re_replica_arrange'] = 0
        self.re_parameters['re_energy_shift'] = 0
        self.its_parameters = {}
        self.its_parameters['its_initialization'] = False
        self.its_parameters['its_replica_arrange'] = 0
        self.its_parameters['its_weight_pk'] = 1e-4
        self.its_parameters['its_energy_shift'] = 0
        self.wall_potential_parameters = {}
        self.wall_potential_parameters['wall_potential'] = False
        self.wall_potential_parameters['wall_number'] = 1
        self.wall_potential_parameters['wall_collective_variable'] = []
        self.wall_potential_parameters['wall_shape'] = []
        self.wall_potential_parameters['wall_type'] = []
        self.wall_potential_parameters['power_wall_direction'] = []
        self.wall_potential_parameters['wall_scaling'] = []
        self.wall_potential_parameters['wall_scope'] = []
        self.wall_potential_parameters['wall_action_atoms'] = []
        self.wall_potential_parameters['planar_wall_point'] = []
        self.wall_potential_parameters['planar_wall_normal_vector'] = []
        self.wall_potential_parameters['spherical_wall_center'] = []
        self.wall_potential_parameters['spherical_wall_radius'] = []
        self.wall_potential_parameters['dot_wall_position'] = []
        for i_parameter in __parameters:
            if len(i_parameter.split()) < 2:
                continue
            ########################## MoREST            ##########################

            elif i_parameter.split()[0].upper() == 'MoREST_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_initialization'] = False
                else:
                    raise Exception('It is not clear whether the MoREST will be initialized.')
                    
            elif i_parameter.split()[0].upper() == 'MoREST_save_parameters_file'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_save_parameters_file'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_save_parameters_file'] = False
                else:
                    raise Exception('It is not clear whether the parameters will be saved in files.')
                    
            elif i_parameter.split()[0].upper() == 'MoREST_load_parameters_file'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_load_parameters_file'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_load_parameters_file'] = False
                else:
                    raise Exception('It is not clear whether the parameters will be loaded from files.')
                    
            elif i_parameter.split()[0].upper() == 'Many_body_potential'.upper():
                self.morest_parameters['many_body_potential'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Input_file'.upper():
                self.morest_parameters['input_file'] = str(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ML_potential_model'.upper():
                self.morest_parameters['ml_potential_model'] = str(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'ML_active_learning'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['ml_active_learning'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['ml_active_learning'] = False

            elif i_parameter.split()[0].upper() == 'ML_energy_uncertainty_tolerance'.upper():
                self.morest_parameters['ml_energy_uncertainty_tolerance'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'ML_appending_set_number'.upper():
                self.morest_parameters['ml_appending_set_number'] = int(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'ML_FD_forces'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['ml_fd_forces'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['ml_fd_forces'] = False  

            elif i_parameter.split()[0].upper() == 'FD_displacement'.upper():
                self.morest_parameters['fd_displacement'] = float(i_parameter.split()[1])

            ########################## Phase space sampling #######################

            elif i_parameter.split()[0].upper() == 'Phase_space_sampling'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['phase_space_sampling'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['phase_space_sampling'] = False
                else:
                    raise Exception('It is not clear whether the sampling method will be used.')

            elif i_parameter.split()[0].upper() == 'Sampling_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['sampling_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['sampling_initialization'] = False
                else:
                    raise Exception('It is not clear whether the sampling method will be initialized.')
                    
            elif i_parameter.split()[0].upper() == 'Sampling_molecule'.upper():
                self.sampling_parameters['sampling_molecule'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_traj_interval'.upper():
                self.sampling_parameters['sampling_traj_interval'] = int(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_method'.upper():
                self.sampling_parameters['sampling_method'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_ensemble'.upper():
                self.sampling_parameters['sampling_ensemble'] = str(i_parameter.split()[1])
                if self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_VR']:
                    self.sampling_parameters['nvt_vr_dt'] = float(i_parameter.split()[2])
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Berendsen'.upper()]:
                    self.sampling_parameters['nvt_berendsen_tau'] = float(i_parameter.split()[2])
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Langevin'.upper()]:
                    self.sampling_parameters['nvt_langevin_gamma'] = float(i_parameter.split()[2])
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_SVR']:
                    self.sampling_parameters['nvt_svr_tau'] = float(i_parameter.split()[2])
                
            ########################## Molecular dynamics #########################
                
            elif i_parameter.split()[0].upper() == 'MD_time_step'.upper():
                self.md_parameters['md_time_step'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'MD_simulation_time'.upper():
                self.md_parameters['md_simulation_time'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'MD_temperature'.upper():
                self.md_parameters['md_temperature'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'MD_clean_rotation'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.md_parameters['md_clean_rotation'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.md_parameters['md_clean_rotation'] = False
                else:
                    raise Exception('It is not clear whether the rotation will be removed.')
                    
            elif i_parameter.split()[0].upper() == 'MD_clean_translation'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.md_parameters['md_clean_translation'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.md_parameters['md_clean_translation'] = False
                else:
                    raise Exception('It is not clear whether the translation will be removed.')
                
            ########################## Trajectory scattering ######################
                
            elif i_parameter.split()[0].upper() == 'Trajectory_scattering'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['trajectory_scattering'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['trajectory_scattering'] = False
                else:
                    raise Exception('It is not clear whether the scattering method will be used.')
                    
            elif i_parameter.split()[0].upper() == 'Scattering_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['scattering_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['scattering_initialization'] = False
                else:
                    raise Exception('It is not clear whether the scattering method will be initialized.')
                    
            elif i_parameter.split()[0].upper() == 'Scattering_pre_thermolized'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['scattering_pre_thermolized'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['scattering_pre_thermolized'] = False
                else:
                    raise Exception('It is not clear whether the pre-thermolized trajectory will be used.')

            elif i_parameter.split()[0].upper() == 'Scattering_traj_number'.upper():
                self.scattering_parameters['scattering_traj_number'] = int(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Scattering_method'.upper():
                self.scattering_parameters['scattering_method'] = str(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_time_step'.upper():
                self.scattering_parameters['scattering_time_step'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_V_collision'.upper():
                self.scattering_parameters['scattering_V_collision'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_E_collision'.upper():
                self.scattering_parameters['scattering_E_collision'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_T_target'.upper():
                self.scattering_parameters['scattering_T_target'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_T_incident'.upper():
                self.scattering_parameters['scattering_T_incident'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_stops_number'.upper():
                self.scattering_parameters['scattering_stops_number'] = int(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_traj_stop'.upper():
                self.traj_stop_parameter = i_parameter.split()[1:]

            elif i_parameter.split()[0].upper() == 'Scattering_traj_length'.upper():
                self.scattering_parameters['scattering_traj_length'] = int(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_target_molecule'.upper():
                self.scattering_parameters['scattering_target_molecule'] = str(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_incident_molecule'.upper():
                self.scattering_parameters['scattering_incident_molecule'] = str(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_R_target'.upper():
                self.scattering_parameters['scattering_R_target'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_R_incident'.upper():
                self.scattering_parameters['scattering_R_incident'] = float(i_parameter.split()[1])


            ########################## Enhanced sampling ##########################

            elif i_parameter.split()[0].upper() == 'Enhanced_sampling'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.enhanced_sampling_parameters['enhanced_sampling'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.enhanced_sampling_parameters['enhanced_sampling'] = False
                else:
                    raise Exception('It is not clear whether the enhanced sampling will be used.')
                
            elif i_parameter.split()[0].upper() == 'Enhanced_sampling_method'.upper():
                self.enhanced_sampling_parameters['enhanced_sampling_method'] = str(i_parameter.split()[1])
            
            ########################## RE parameters  #############################

            elif i_parameter.split()[0].upper() == 'RE_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.re_parameters['re_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.re_parameters['re_initialization'] = False
                else:
                    raise Exception('It is not clear whether the replica exchange sampling will be initialized.')
                
            elif i_parameter.split()[0].upper() == 'RE_lower_bound_temperature'.upper():
                self.re_parameters['re_lower_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_upper_bound_temperature'.upper():
                self.re_parameters['re_upper_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_number_of_replica'.upper():
                self.re_parameters['re_number_of_replica'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_replica_arrange'.upper():
                self.re_parameters['re_replica_arrange'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_replica_temperatures'.upper():
                tmp_temperatures = []
                for i in range(self.re_parameters['re_number_of_replica']):
                    tmp_temperatures.append(float(i_parameter.split()[i+1]))
                self.re_parameters['re_replica_temperatures'] = np.array(tmp_temperatures)
                
            elif i_parameter.split()[0].upper() == 'RE_swap_interval'.upper():
                self.re_parameters['re_swap_interval'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_init_structures_list'.upper():
                self.re_parameters['re_init_structures_list'] = str(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'RE_energy_shift'.upper():
                self.re_parameters['re_energy_shift'] = float(i_parameter.split()[1])

            ########################## ITS parameters    ##########################

            elif i_parameter.split()[0].upper() == 'ITS_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.its_parameters['its_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.its_parameters['its_initialization'] = False
                else:
                    raise Exception('It is not clear whether the integrated tempering sampling will be initialized.')
                
            elif i_parameter.split()[0].upper() == 'ITS_lower_bound_temperature'.upper():
                self.its_parameters['its_lower_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_upper_bound_temperature'.upper():
                self.its_parameters['its_upper_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_number_of_replica'.upper():
                self.its_parameters['its_number_of_replica'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_replica_arrange'.upper():
                self.its_parameters['its_replica_arrange'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_replica_temperatures'.upper():
                tmp_temperatures = []
                for i in range(self.its_parameters['its_number_of_replica']):
                    tmp_temperatures.append(float(i_parameter.split()[i+1]))
                self.its_parameters['its_replica_temperatures'] = np.array(tmp_temperatures)
                
            elif i_parameter.split()[0].upper() == 'ITS_initial_nk'.upper():
                tmp_nk = []
                for i in range(self.its_parameters['its_number_of_replica']):
                    tmp_nk.append(float(i_parameter.split()[i+1]))
                self.its_parameters['its_initial_nk'] = np.array(tmp_nk)
                
            elif i_parameter.split()[0].upper() == 'ITS_pk0'.upper():
                tmp_pk0 = []
                for i in range(self.its_parameters['its_number_of_replica']):
                    tmp_pk0.append(float(i_parameter.split()[i+1]))
                self.its_parameters['its_pk0'] = np.array(tmp_pk0)
                
            elif i_parameter.split()[0].upper() == 'ITS_trial_MD_steps'.upper():
                self.its_parameters['its_trial_MD_steps'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_delta_pk'.upper():
                self.its_parameters['its_delta_pk'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_weight_pk'.upper():
                self.its_parameters['its_weight_pk'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_energy_shift'.upper():
                self.its_parameters['its_energy_shift'] = float(i_parameter.split()[1])
            
            ########################## Wall potential #############################

            elif i_parameter.split()[0].upper() == 'Wall_potential'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['wall_potential'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['wall_potential'] = False
                else:
                    raise Exception('It is not clear whether the wall potential will be used.')
                
            elif i_parameter.split()[0].upper() == 'Wall_number'.upper():
                self.wall_potential_parameters['wall_number'] = int(i_parameter.split()[1])
   
            elif i_parameter.split()[0].upper() == 'Wall_collective_variable'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['wall_collective_variable'].append(True)
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['wall_collective_variable'].append(False)
                else:
                    raise Exception('It is not clear whether the collective variable will be used.')
                
            elif i_parameter.split()[0].upper() == 'Wall_shape'.upper():
                self.wall_potential_parameters['wall_shape'].append(str(i_parameter.split()[1]).lower())
                
            elif i_parameter.split()[0].upper() == 'Wall_type'.upper():
                self.wall_potential_parameters['wall_type'].append(str(i_parameter.split()[1]).lower())
                if i_parameter.split()[1].upper() in ['power_wall'.upper()]:
                    self.wall_potential_parameters['power_wall_direction'].append(np.sign(int(i_parameter.split()[2])))
                else:
                    self.wall_potential_parameters['power_wall_direction'].append(0)
                
            elif i_parameter.split()[0].upper() == 'Wall_scaling'.upper():
                self.wall_potential_parameters['wall_scaling'].append(float(i_parameter.split()[1]))
                
            elif i_parameter.split()[0].upper() == 'Wall_scope'.upper():
                self.wall_potential_parameters['wall_scope'].append(float(i_parameter.split()[1]))
                
            elif i_parameter.split()[0].upper() == 'Wall_action_atoms'.upper():
                self.wall_potential_parameters['wall_action_atoms'].append(i_parameter.split()[1:])
                
            ########################## Planar wall #################################

            elif i_parameter.split()[0].upper() == 'Planar_wall_point'.upper():
                tmp_wall_point = []
                for i in range(3):
                    tmp_wall_point.append(float(i_parameter.split()[i+1]))
                self.wall_potential_parameters['planar_wall_point'].append(np.array(tmp_wall_point))
            
            elif i_parameter.split()[0].upper() == 'Planar_wall_normal_vector'.upper():
                tmp_wall_normal_vector = []
                for i in range(3):
                    tmp_wall_normal_vector.append(float(i_parameter.split()[i+1]))
                tmp_wall_normal_vector = np.array(tmp_wall_normal_vector)
                self.wall_potential_parameters['planar_wall_normal_vector'].append(tmp_wall_normal_vector / np.linalg.norm(tmp_wall_normal_vector))
                
            ########################## Spherical wall #############################

            elif i_parameter.split()[0].upper() == 'Spherical_wall_center'.upper():
                tmp_wall_center = []
                for i in range(3):
                    tmp_wall_center.append(float(i_parameter.split()[i+1]))
                self.wall_potential_parameters['spherical_wall_center'].append(np.array(tmp_wall_center))
            
            elif i_parameter.split()[0].upper() == 'Spherical_wall_radius'.upper():
                self.wall_potential_parameters['spherical_wall_radius'].append(float(i_parameter.split()[1]))

            ########################## dot wall ###################################

            elif i_parameter.split()[0].upper() == 'Dot_wall_position'.upper():
                tmp_wall_center = []
                for i in range(3):
                    tmp_wall_center.append(float(i_parameter.split()[i+1]))
                self.wall_potential_parameters['dot_wall_position'].append(np.array(tmp_wall_center))
                
    def write_parameters(self, log_morest):
        log_morest.write('\n')
        try:
            for key in self.morest_parameters:
                log_morest.write(key+' : '+str(self.morest_parameters[key])+'\n')
            log_morest.write('\n')
        except:
            pass
        try:
            if self.sampling_parameters['phase_space_sampling']:
                for key in self.sampling_parameters:
                    log_morest.write(key+' : '+str(self.sampling_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.sampling_parameters['phase_space_sampling']:
                for key in self.md_parameters:
                    log_morest.write(key+' : '+str(self.md_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.scattering_parameters['trajectory_scattering']:
                for key in self.scattering_parameters:
                    log_morest.write(key+' : '+str(self.scattering_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.enhanced_sampling_parameters:
                    log_morest.write(key+' : '+str(self.enhanced_sampling_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.its_parameters:
                    log_morest.write(key+' : '+str(self.its_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.re_parameters:
                    log_morest.write(key+' : '+str(self.re_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.wall_potential_parameters:
                    log_morest.write(key+' : '+str(self.wall_potential_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.planar_wall_parameters:
                    log_morest.write(key+' : '+str(self.planar_wall_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.spherical_wall_parameters:
                    log_morest.write(key+' : '+str(self.spherical_wall_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.dot_wall_parameters:
                    log_morest.write(key+' : '+str(self.dot_wall_parameters[key])+'\n')
                log_morest.write('\n')
        except:
            pass
        log_morest.write('\n')
            

                
    def get_morest_parameters(self, log_morest=None):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_morest_parameters.npy', self.morest_parameters)
        self.morest_parameters['phase_space_sampling'] = self.sampling_parameters['phase_space_sampling']
        self.morest_parameters['trajectory_scattering'] = self.scattering_parameters['trajectory_scattering']
        self.morest_parameters['enhanced_sampling'] = self.enhanced_sampling_parameters['enhanced_sampling']
        self.morest_parameters['wall_potential'] = self.wall_potential_parameters['wall_potential']
        if type(log_morest) != type(None):
            for key in self.morest_parameters:
                log_morest.write(key+' : '+str(self.morest_parameters[key])+'\n')
            log_morest.write('\n')
        return self.morest_parameters

    def get_sampling_parameters(self, log_morest=None):
        if self.morest_parameters['morest_initialization'] == True:
           self.sampling_parameters['sampling_initialization'] = True
        if self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Berendsen'.upper()]:
            self.sampling_parameters['nvt_berendsen_tau'] *= units.fs
        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Langevin'.upper()]:
            self.sampling_parameters['nvt_langevin_gamma'] /= units.fs
        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_SVR']:
            self.sampling_parameters['nvt_svr_tau'] *= units.fs
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_sampling_parameters.npy', self.sampling_parameters)
        if type(log_morest) != type(None):
            for key in self.sampling_parameters:
                if key in ['nvt_berendsen_tau','nvt_svr_tau']:
                    log_morest.write(key+' : '+str(self.sampling_parameters[key]/units.fs)+'\n')
                elif key in ['nvt_langevin_gamma']:
                    log_morest.write(key+' : '+str(self.sampling_parameters[key]*units.fs)+'\n')
                else:
                    log_morest.write(key+' : '+str(self.sampling_parameters[key])+'\n')
            log_morest.write('\n')
        return self.sampling_parameters
    
    def get_md_parameters(self, log_morest=None):
        self.md_parameters['md_time_step'] *= units.fs
        self.md_parameters['md_simulation_time'] *= units.fs
        if self.sampling_parameters['phase_space_sampling']:
            if self.morest_parameters['morest_save_parameters_file']:
                np.save('MoREST_MD_parameters.npy', self.md_parameters)
            if type(log_morest) != type(None):
                for key in self.md_parameters:
                    if key in ['md_time_step','md_simulation_time']:
                        log_morest.write(key+' : '+str(self.md_parameters[key]/units.fs)+'\n')
                    else:
                        log_morest.write(key+' : '+str(self.md_parameters[key])+'\n')
                log_morest.write('\n')
        return self.md_parameters

    def get_scattering_parameters(self, log_morest=None):
        if self.morest_parameters['morest_initialization'] == True:
           self.scattering_parameters['scattering_initialization'] = True
        self.scattering_parameters['scattering_time_step'] *= units.fs
        self.scattering_parameters['scattering_V_collision'] /= units.fs
        if self.scattering_parameters['scattering_stops_number'] == 0:
            self.scattering_parameters['scattering_traj_stop'] = None
        else:
            traj_stop_CVs = []
            i_loc = 0 # used to locate the index of the CVs parameters
            for i_stop in range(self.scattering_parameters['scattering_stops_number']):
                if self.traj_stop_parameter[0+i_loc].upper() == 'None'.upper():
                    self.scattering_parameters['scattering_traj_stop'] = None
                    break
                elif self.traj_stop_parameter[0+i_loc].upper() == 'distance'.upper():
                    tmp_stop = []
                    tmp_stop.append('distance')
                    tmp_stop.append(int(self.traj_stop_parameter[1+i_loc]))
                    tmp_stop.append(float(self.traj_stop_parameter[2+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[3+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[4+i_loc]))
                    i_loc += 5
                    traj_stop_CVs.append(tmp_stop)
                elif self.traj_stop_parameter[0+i_loc].upper() == 'angle'.upper():
                    tmp_stop = []
                    tmp_stop.append('angle')
                    tmp_stop.append(int(self.traj_stop_parameter[1+i_loc]))
                    tmp_stop.append(float(self.traj_stop_parameter[2+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[3+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[4+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[5+i_loc]))
                    i_loc += 6
                    traj_stop_CVs.append(tmp_stop)
                elif self.traj_stop_parameter[0+i_loc].upper() == 'dihedral'.upper():
                    tmp_stop = []
                    tmp_stop.append('dihedral')
                    tmp_stop.append(int(self.traj_stop_parameter[1+i_loc]))
                    tmp_stop.append(float(self.traj_stop_parameter[2+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[3+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[4+i_loc]))
                    tmp_stop.append(int(self.traj_stop_parameter[6+i_loc]))
                    i_loc += 7
                    traj_stop_CVs.append(tmp_stop)
                elif self.traj_stop_parameter[0+i_loc].upper() == 'central_R_one'.upper():
                    tmp_stop = []
                    tmp_stop.append('central_R_one')
                    tmp_stop.append(int(self.traj_stop_parameter[1+i_loc]))
                    tmp_stop.append(float(self.traj_stop_parameter[2+i_loc]))
                    N_check = self.traj_stop_parameter[3+i_loc]
                    if N_check.upper() == 'all'.upper():
                        tmp_stop.append(N_check)
                        i_loc += 4
                        traj_stop_CVs.append(tmp_stop)
                    else:
                        N_check = int(N_check)
                        tmp_stop.append(N_check)
                        atom_list = []
                        for i_atom in range(N_check):
                            atom_list.append(int(self.traj_stop_parameter[4+i_loc+i_atom]))
                        tmp_stop.append(atom_list)
                        i_loc += 4+N_check
                        traj_stop_CVs.append(tmp_stop)
                elif self.traj_stop_parameter[0+i_loc].upper() == 'central_R_all'.upper():
                    tmp_stop = []
                    tmp_stop.append('central_R_all')
                    tmp_stop.append(int(self.traj_stop_parameter[1+i_loc]))
                    tmp_stop.append(float(self.traj_stop_parameter[2+i_loc]))
                    N_check = self.traj_stop_parameter[3+i_loc]
                    if N_check.upper() == 'all'.upper():
                        tmp_stop.append(N_check)
                        i_loc += 4
                        traj_stop_CVs.append(tmp_stop)
                    else:
                        N_check = int(N_check)
                        tmp_stop.append(N_check)
                        atom_list = []
                        for i_atom in range(N_check):
                            atom_list.append(int(self.traj_stop_parameter[4+i_loc+i_atom]))
                        tmp_stop.append(atom_list)
                        i_loc += 4+N_check
                        traj_stop_CVs.append(tmp_stop)
                else:
                    raise Exception('It is not clear which stop condition will be used.')
            self.scattering_parameters['scattering_traj_stop'] = traj_stop_CVs
            #print(type(self.scattering_parameters['scattering_traj_stop'])) # DEBUG
            #print(self.scattering_parameters['scattering_traj_stop'])       # DEBUG
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_scattering_parameters.npy', self.scattering_parameters)
        if type(log_morest) != type(None):
            for key in self.scattering_parameters:
                if key in ['scattering_time_step']:
                    log_morest.write(key+' : '+str(self.scattering_parameters[key]/units.fs)+'\n')
                elif key in ['scattering_V_collision']:
                    log_morest.write(key+' : '+str(self.scattering_parameters[key]*units.fs)+'\n')
                else:
                    log_morest.write(key+' : '+str(self.scattering_parameters[key])+'\n')
            log_morest.write('\n')
        return self.scattering_parameters

    def get_enhanced_sampling_parameters(self, log_morest=None):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_enhanced_sampling_parameters.npy', self.enhanced_sampling_parameters)
        if type(log_morest) != type(None):
            for key in self.enhanced_sampling_parameters:
                log_morest.write(key+' : '+str(self.enhanced_sampling_parameters[key])+'\n')
            log_morest.write('\n')
        return self.enhanced_sampling_parameters
        
    def get_re_parameters(self, log_morest=None):
        if self.morest_parameters['morest_initialization'] == True:
           self.re_parameters['re_initialization'] = True
        elif  self.morest_parameters['morest_initialization'] == False:
           self.re_parameters['re_initialization'] = False
        if not 're_replica_temperatures' in self.re_parameters:
            if int(self.re_parameters['re_replica_arrange']) == -1:
                replica_temperatures = np.linspace(self.re_parameters['re_lower_bound_temperature'],\
                                                  self.re_parameters['re_upper_bound_temperature'],\
                                                  num=self.re_parameters['re_number_of_replica'],\
                                                  endpoint=True)
            elif int(self.re_parameters['re_replica_arrange']) == 0:
                replica_temperatures = np.geomspace(self.re_parameters['re_lower_bound_temperature'],\
                                                   self.re_parameters['re_upper_bound_temperature'],\
                                                   num=self.re_parameters['re_number_of_replica'],\
                                                   endpoint=True)
            else:
                raise Exception('No RE_replica_arrange type was matched.')
            self.re_parameters['re_replica_temperatures'] = replica_temperatures
        
        self.re_parameters['re_replica_beta'] = 1/(self.re_parameters['re_replica_temperatures'] * units.kB)
        #                                            scipy.constants.value('Boltzmann constant in eV/K'))
        if not 're_init_structures_list' in self.re_parameters:
            self.re_parameters['re_multiple_initi_structures'] = False
        else:
            self.re_parameters['re_multiple_initi_structures'] = True
        #if self.re_parameters['re_initialization']:
        #    self.re_parameters['re_current_swap_step'] = 0
        #    self.re_parameters['re_current_replica'] = 0
        #else:
        #    step_replica = np.loadtxt('MoREST_RE_current_step_replica.log')
        #    self.re_parameters['re_current_swap_step'] = step_replica[0]
        #    self.re_parameters['re_current_replica'] = step_replica[1]
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_RE_parameters.npy',self.re_parameters)
        if type(log_morest) != type(None):
            for key in self.re_parameters:
                log_morest.write(key+' : '+str(self.re_parameters[key])+'\n')
            log_morest.write('\n')
        return self.re_parameters
    
    def get_its_parameters(self, log_morest=None):
        if self.morest_parameters['morest_initialization'] == True:
           self.its_parameters['its_initialization'] = True
        elif  self.morest_parameters['morest_initialization'] == False:
           self.its_parameters['its_initialization'] = False
        self.its_parameters['its_criteria_pk'] = self.its_parameters['its_delta_pk'] / \
                                                self.its_parameters['its_number_of_replica']
        if not 'its_replica_temperatures' in self.its_parameters:
            if int(self.its_parameters['its_replica_arrange']) == -1:
                replica_temperatures = np.linspace(self.its_parameters['its_lower_bound_temperature'],\
                                                  self.its_parameters['its_upper_bound_temperature'],\
                                                  num=self.its_parameters['its_number_of_replica'],\
                                                  endpoint=True)
            elif int(self.its_parameters['its_replica_arrange']) == 0:
                replica_temperatures = np.geomspace(self.its_parameters['its_lower_bound_temperature'],\
                                                   self.its_parameters['its_upper_bound_temperature'],\
                                                   num=self.its_parameters['its_number_of_replica'],\
                                                   endpoint=True)
            else:
                raise Exception('No ITS_replica_arrange type was matched.')
            self.its_parameters['its_replica_temperatures'] = replica_temperatures
        
        self.its_parameters['its_replica_beta'] = 1/(self.its_parameters['its_replica_temperatures'] * units.kB)
        #                                            scipy.constants.value('Boltzmann constant in eV/K')
        if not 'its_initial_nk' in self.its_parameters:
            #self.its_parameters['its_initial_nk'] = np.exp(self.its_parameters['its_replica_beta'])
            #self.its_parameters['its_initial_nk'] = np.exp(-1*self.its_parameters['its_replica_temperatures'])
            self.its_parameters['its_initial_nk'] = self.its_parameters['its_replica_beta'] /\
                                                    np.sum(self.its_parameters['its_replica_beta'])
        if not 'its_pk0' in self.its_parameters:
            self.its_parameters['its_pk0'] = np.ones((self.its_parameters['its_number_of_replica'])) /\
                                                   self.its_parameters['its_number_of_replica']
        #with open('MoREST_ITS_parameters.json','w') as its_json:
        #    json.dump(self.its_parameters,its_json, cls=NumpyArrayEncoder)
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_ITS_parameters.npy',self.its_parameters)
        if type(log_morest) != type(None):
            for key in self.its_parameters:
                log_morest.write(key+' : '+str(self.its_parameters[key])+'\n')
            log_morest.write('\n')
        return self.its_parameters
            
    def get_wall_potential_parameters(self, log_morest=None):
        for key in ['wall_collective_variable', 'wall_shape', 'wall_type', 'power_wall_direction', \
            'wall_scaling', 'wall_scope', 'wall_action_atoms', \
            'planar_wall_point', 'planar_wall_normal_vector', \
            'spherical_wall_center', 'spherical_wall_radius', \
            'dot_wall_position']:
            self.wall_potential_parameters[key] = self.wall_potential_parameters[key]\
                [:self.wall_potential_parameters['wall_number']]
        for i_wall,i_wall_type in enumerate(self.wall_potential_parameters['wall_type']):
            if i_wall_type.upper() in ['power_wall'.upper()]:
                try:
                    self.wall_potential_parameters['wall_scope'][i_wall] >= 1
                except:
                    if type(log_morest) != type(None):
                        log_morest.write('Parameter wall_scope should be >= 1 for power potential.\n')
                        log_morest.close()
                    raise Exception('Parameter wall_scope should be >= 1 for power potential.')
            elif not i_wall_type.upper() in ['opaque_wall'.upper(), 'translucent_wall'.upper(), 'power_wall'.upper()]:
                log_morest.write('Wall type is not recognized.\n')
                log_morest.close()
                raise Exception('Wall type is not recognized.')
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_wall_potential_parameters.npy', self.wall_potential_parameters)
        if type(log_morest) != type(None):
            for key in self.wall_potential_parameters:
                log_morest.write(key+' : '+str(self.wall_potential_parameters[key])+'\n')
            log_morest.write('\n')
        return self.wall_potential_parameters
    
