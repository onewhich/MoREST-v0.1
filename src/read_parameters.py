import sys
import numpy as np
import scipy.constants
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
        log_morest:     log file for MoREST.
        parameter_file: parameter file name. Default name is 'MoREST.parameter' in current directory.
    --------
    Enhanced sampling parameters are stored in dictionary named 'enhanced_sampling_parameters'.
    ITS parameters are stored in dictionary named 'its_parameters'.
    Wall potential parameters are stored in dictionary named 'wall_potential_parameters'.
    Plane wall parameters are stored in dictionary named 'plane_wall_parameters'.
    '''
    
    def __init__(self, log_morest, parameter_file='MoREST.in'):
        self.__log_morest = log_morest
        try:
            __parameters = open(parameter_file,'r').readlines()
        except FileNotFoundError:
            self.__log_morest.write('Can not open parameter file: '+str(parameter_file)+'\n')
            self.__log_morest.close()
            sys.exit(0)
        
        self.morest_parameters = {}
        self.morest_parameters['morest_initialization'] = True
        self.morest_parameters['morest_save_parameters_file'] = False
        self.morest_parameters['morest_load_parameters_file'] = False
        self.morest_parameters['fd_displacement'] = 0.0025
        self.sampling_parameters = {}
        self.sampling_parameters['phase_space_sampling'] = False
        self.sampling_parameters['sampling_initialization'] = True
        self.sampling_parameters['sampling_molecule'] = 'MoREST.str'
        self.md_parameters = {}
        self.md_parameters['md_clean_rotation'] = True
        self.md_parameters['md_clean_translation'] = True
        self.scattering_parameters = {}
        self.scattering_parameters['trajectory_scattering'] = False
        self.scattering_parameters['scattering_initialization'] = True
        self.scattering_parameters['scattering_pre_thermolized'] = False
        self.scattering_parameters['scattering_traj_stop'] = None
        self.scattering_parameters['scattering_traj_length'] = None
        self.scattering_parameters['scattering_target_molecule'] = 'MoREST.str_target'
        self.scattering_parameters['scattering_incident_molecule'] = 'MoREST.str_incident'
        self.enhanced_sampling_parameters = {}
        self.enhanced_sampling_parameters['enhanced_sampling'] = False
        self.its_parameters = {}
        self.its_parameters['its_initialization'] = True
        self.re_parameters = {}
        self.re_parameters['re_initialization'] = True
        self.wall_potential_parameters = {}
        self.wall_potential_parameters['wall_potential'] = False
        self.wall_potential_parameters['collective_variable'] = False
        self.plane_wall_parameters = {}
        self.spherical_wall_parameters = {}
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
                    self.__log_morest.write('It is not clear whether the MoREST will be initialized.\n')
                    self.__log_morest.close()
                    raise Exception('Will you initialize the MoREST or not?')
                    
            elif i_parameter.split()[0].upper() == 'MoREST_save_parameters_file'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_save_parameters_file'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_save_parameters_file'] = False
                    
            elif i_parameter.split()[0].upper() == 'MoREST_load_parameters_file'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_load_parameters_file'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_load_parameters_file'] = False
                    
            elif i_parameter.split()[0].upper() == 'Many_body_potential'.upper():
                self.morest_parameters['many_body_potential'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Input_file'.upper():
                self.morest_parameters['input_file'] = str(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ML_potential_model'.upper():
                self.morest_parameters['ml_potential_model'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'FD_displacement'.upper():
                self.morest_parameters['fd_displacement'] = float(i_parameter.split()[1])

            ########################## Phase space sampling #######################

            elif i_parameter.split()[0].upper() == 'Phase_space_sampling'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['phase_space_sampling'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['phase_space_sampling'] = False
                else:
                    self.__log_morest.write('It is not clear whether the sampling method will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use sampling method or not?')
                    
            elif i_parameter.split()[0].upper() == 'Sampling_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['sampling_initialization'] = True
                    # change MoREST_initialization as True
                    self.morest_parameters['morest_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['sampling_initialization'] = False
                    
            elif i_parameter.split()[0].upper() == 'Sampling_molecule'.upper():
                self.sampling_parameters['sampling_molecule'] = i_parameter.split()[1]
                    
            elif i_parameter.split()[0].upper() == 'Sampling_traj_interval'.upper():
                self.sampling_parameters['sampling_traj_interval'] = int(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_method'.upper():
                self.sampling_parameters['sampling_method'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_ensemble'.upper():
                self.sampling_parameters['sampling_ensemble'] = str(i_parameter.split()[1])
                if self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_VR']:
                    self.sampling_parameters['nvt_vr_dt'] = float(i_parameter.split()[2])
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
                    
            elif i_parameter.split()[0].upper() == 'MD_clean_translation'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.md_parameters['md_clean_translation'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.md_parameters['md_clean_translation'] = False
                
            ########################## Trajectory scattering ######################
                
            elif i_parameter.split()[0].upper() == 'Trajectory_scattering'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['trajectory_scattering'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['trajectory_scattering'] = False
                else:
                    self.__log_morest.write('It is not clear whether the scattering method will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use scattering method or not?')
                    
            elif i_parameter.split()[0].upper() == 'Scattering_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['scattering_initialization'] = True
                    self.morest_parameters['morest_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['scattering_initialization'] = False
                    
            elif i_parameter.split()[0].upper() == 'Scattering_pre_thermolized'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.scattering_parameters['scattering_pre_thermolized'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.scattering_parameters['scattering_pre_thermolized'] = False

            elif i_parameter.split()[0].upper() == 'Scattering_traj_number'.upper():
                self.scattering_parameters['scattering_traj_number'] = int(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Scattering_method'.upper():
                self.scattering_parameters['scattering_method'] = str(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_temperature'.upper():
                self.scattering_parameters['scattering_temperature'] = float(i_parameter.split()[1])

            elif i_parameter.split()[0].upper() == 'Scattering_time_step'.upper():
                self.scattering_parameters['scattering_time_step'] = float(i_parameter.split()[1])

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
                    self.__log_morest.write('It is not clear whether the enhanced sampling will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use enhanced sampling or not?')
                
            elif i_parameter.split()[0].upper() == 'Enhanced_sampling_method'.upper():
                self.enhanced_sampling_parameters['enhanced_sampling_method'] = str(i_parameter.split()[1])
            
            ########################## ITS parameters    ##########################

            elif i_parameter.split()[0].upper() == 'ITS_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.its_parameters['its_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.its_parameters['its_initialization'] = False
                
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
                
            elif i_parameter.split()[0].upper() == 'ITS_energy_shift'.upper():
                self.its_parameters['its_energy_shift'] = float(i_parameter.split()[1])
            
            ########################## RE parameters  #############################

            elif i_parameter.split()[0].upper() == 'RE_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.re_parameters['re_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.re_parameters['re_initialization'] = False
                
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
                
            elif i_parameter.split()[0].upper() == 'RE_energy_shift'.upper():
                self.re_parameters['re_energy_shift'] = float(i_parameter.split()[1])

            ########################## Wall potential #############################

            elif i_parameter.split()[0].upper() == 'Wall_potential'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['wall_potential'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['wall_potential'] = False
                else:
                    self.__log_morest.write('It is not clear whether the wall potential will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use wall potential or not?')
                    
            elif i_parameter.split()[0].upper() == 'Collective_variable'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['collective_variable'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['collective_variable'] = False
                else:
                    self.__log_morest.write('It is not clear whether the collective variable will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use collective variable or not?')
                
            elif i_parameter.split()[0].upper() == 'Wall_type'.upper():
                self.wall_potential_parameters['wall_type'] = str(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'Wall_scaling'.upper():
                self.wall_potential_parameters['wall_scaling'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'Wall_scope'.upper():
                self.wall_potential_parameters['wall_scope'] = float(i_parameter.split()[1])
                
            ########################## Plane wall #################################

            elif i_parameter.split()[0].upper() == 'Plane_wall_point'.upper():
                tmp_wall_point = []
                for i in range(3):
                    tmp_wall_point.append(float(i_parameter.split()[i+1]))
                self.plane_wall_parameters['plane_wall_point'] = np.array(tmp_wall_point)
            
            elif i_parameter.split()[0].upper() == 'Plane_wall_normal_vector'.upper():
                tmp_wall_normal_vector = []
                for i in range(3):
                    tmp_wall_normal_vector.append(float(i_parameter.split()[i+1]))
                tmp_wall_normal_vector = np.array(tmp_wall_normal_vector)
                self.plane_wall_parameters['plane_wall_normal_vector'] = tmp_wall_normal_vector / np.linalg.norm(tmp_wall_normal_vector)
                
            ########################## Spherical wall #############################

            elif i_parameter.split()[0].upper() == 'Spherical_wall_center'.upper():
                tmp_wall_center = []
                for i in range(3):
                    tmp_wall_center.append(float(i_parameter.split()[i+1]))
                self.spherical_wall_parameters['spherical_wall_center'] = np.array(tmp_wall_center)
            
            elif i_parameter.split()[0].upper() == 'Spherical_wall_radius'.upper():
                self.spherical_wall_parameters['spherical_wall_radius'] = float(i_parameter.split()[1])
                
    def write_parameters(self):
        self.__log_morest.write('\n')
        try:
            for key in self.morest_parameters:
                self.__log_morest.write(key+' : '+str(self.morest_parameters[key])+'\n')
        except:
            pass
        try:
            if self.sampling_parameters['phase_space_sampling']:
                for key in self.sampling_parameters:
                    self.__log_morest.write(key+' : '+str(self.sampling_parameters[key])+'\n')
        except:
            pass
        try:
            if self.sampling_parameters['phase_space_sampling']:
                for key in self.md_parameters:
                    self.__log_morest.write(key+' : '+str(self.md_parameters[key])+'\n')
        except:
            pass
        try:
            if self.scattering_parameters['trajectory_scattering']:
                for key in self.scattering_parameters:
                    self.__log_morest.write(key+' : '+str(self.scattering_parameters[key])+'\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.enhanced_sampling_parameters:
                    self.__log_morest.write(key+' : '+str(self.enhanced_sampling_parameters[key])+'\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.its_parameters:
                    self.__log_morest.write(key+' : '+str(self.its_parameters[key])+'\n')
        except:
            pass
        try:
            if self.enhanced_sampling_parameters['enhanced_sampling']:
                for key in self.re_parameters:
                    self.__log_morest.write(key+' : '+str(self.re_parameters[key])+'\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.wall_potential_parameters:
                    self.__log_morest.write(key+' : '+str(self.wall_potential_parameters[key])+'\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.plane_wall_parameters:
                    self.__log_morest.write(key+' : '+str(self.plane_wall_parameters[key])+'\n')
        except:
            pass
        try:
            if self.wall_potential_parameters['wall_potential']:
                for key in self.spherical_wall_parameters:
                    self.__log_morest.write(key+' : '+str(self.spherical_wall_parameters[key])+'\n')
        except:
            pass
        self.__log_morest.write('\n')
            

                
    def get_morest_parameters(self):
        #if self.morest_parameters['morest_save_parameters_file']:
        #    np.save('MoREST_morest_parameters.npy', self.morest_parameters)
        return self.morest_parameters

    def get_sampling_parameters(self):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_sampling_parameters.npy', self.sampling_parameters)
        return self.sampling_parameters
    
    def get_md_parameters(self):
        self.md_parameters['md_time_step'] = self.md_parameters['md_time_step'] * units.fs
        self.md_parameters['md_simulation_time'] = self.md_parameters['md_simulation_time'] * units.fs
        if self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_SVR']:
            self.sampling_parameters['nvt_svr_tau'] = self.sampling_parameters['nvt_svr_tau'] * units.fs
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_MD_parameters.npy', self.md_parameters)
        return self.md_parameters

    def get_scattering_parameters(self):
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
                    self.__log_morest.write('It is not clear which stop condition will be used.\n')
                    self.__log_morest.close()
                    raise Exception('Will you use stop condition or not?')
            self.scattering_parameters['scattering_traj_stop'] = traj_stop_CVs
            #print(type(self.scattering_parameters['scattering_traj_stop'])) # DEBUG
            #print(self.scattering_parameters['scattering_traj_stop'])       # DEBUG
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_scattering_parameters.npy', self.scattering_parameters)
        return self.scattering_parameters

    def get_enhanced_sampling_parameters(self):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_enhanced_sampling_parameters.npy', self.enhanced_sampling_parameters)
        return self.enhanced_sampling_parameters
        
    def get_its_parameters(self):
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
                self.__log_morest.write('No ITS_replica_arrange type was matched.\n')
                self.__log_morest.close()
                raise Exception('No ITS_replica_arrange type was matched.')
            self.its_parameters['its_replica_temperatures'] = replica_temperatures
        
        self.its_parameters['its_replica_beta'] = 1/(self.its_parameters['its_replica_temperatures'] *\
                                                    scipy.constants.value('Boltzmann constant in eV/K'))
        try:
            self.its_parameters['its_initial_nk'] = np.loadtxt('MoREST_ITS_nk.npy')
        except:
            pass
        if not 'its_initial_nk' in self.its_parameters:
            self.its_parameters['its_initial_nk'] = np.exp(self.its_parameters['its_replica_beta'])
            self.its_parameters['its_initial_nk'] = self.its_parameters['its_initial_nk'] /\
                                                    np.max(self.its_parameters['its_initial_nk'])
        try:
            self.its_parameters['its_pk0'] = np.loadtxt('MoREST_ITS_pk.npy')
        except:
            pass
        if not 'its_pk0' in self.its_parameters:
            self.its_parameters['its_pk0'] = np.ones((self.its_parameters['its_number_of_replica'])) /\
                                                   self.its_parameters['its_number_of_replica']
        #with open('MoREST_ITS_parameters.json','w') as its_json:
        #    json.dump(self.its_parameters,its_json, cls=NumpyArrayEncoder)
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_ITS_parameters.npy',self.its_parameters)
        return self.its_parameters
            
    def get_re_parameters(self):
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
                self.__log_morest.write('No RE_replica_arrange type was matched.\n')
                self.__log_morest.close()
                raise Exception('No RE_replica_arrange type was matched.')
            self.re_parameters['re_replica_temperatures'] = replica_temperatures
        
        self.re_parameters['re_replica_beta'] = 1/(self.re_parameters['re_replica_temperatures'] *\
                                                    scipy.constants.value('Boltzmann constant in eV/K'))
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_RE_parameters.npy',self.re_parameters)
        return self.re_parameters
    
    def get_wall_potential_parameters(self):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_wall_potential_parameters.npy', self.wall_potential_parameters)
        return self.wall_potential_parameters
    
    def get_plane_wall_parameters(self):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_plane_wall_parameters.npy', self.plane_wall_parameters)
        return self.plane_wall_parameters

    def get_spherical_wall_parameters(self):
        if self.morest_parameters['morest_save_parameters_file']:
            np.save('MoREST_spherical_wall_parameters.npy', self.spherical_wall_parameters)
        return self.spherical_wall_parameters
    
