import os,sys
import numpy as np
import scipy.constants
import json
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
        __log_morest = log_morest
        try:
            __parameters = open(parameter_file,'r').readlines()
        except FileNotFoundError:
            __log_morest.write('Can not open parameter file: '+str(parameter_file)+'\n')
            __log_morest.close()
            sys.exit(0)
        
        self.morest_parameters = {}
        self.morest_parameters['morest_initialization'] = True
        self.morest_parameters['morest_api_fortran'] = False
        self.sampling_parameters = {}
        self.sampling_parameters['phase_space_sampling'] = False
        self.sampling_parameters['sampling_restart'] = False
        self.sampling_parameters['sampling_clean_rotation'] = True
        self.sampling_parameters['sampling_clean_translation'] = True
        self.sampling_parameters['fd_displacement'] = 0.0025
        self.md_parameters = {}
        self.enhanced_sampling_parameters = {}
        self.enhanced_sampling_parameters['enhanced_sampling'] = False
        self.its_parameters = {}
        self.its_parameters['its_initialization'] = True
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
                    __log_morest.write('It is not clear whether the MoREST will be initialized.\n')
                    __log_morest.close()
                    raise Exception('Will you initialize the MoREST or not?')

            elif i_parameter.split()[0].upper() == 'MoREST_API_Fortran'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.morest_parameters['morest_api_fortran'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.morest_parameters['morest_api_fortran'] = False
                else:
                    __log_morest.write('It is not clear whether the MoREST API for Fortran code will be used.\n')
                    __log_morest.close()
                    raise Exception('Will you use the MoREST API for Fortran code or not?')

            ########################## Phase space sampling #######################
            elif i_parameter.split()[0].upper() == 'Phase_space_sampling'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['phase_space_sampling'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['phase_space_sampling'] = False
                else:
                    __log_morest.write('It is not clear whether the sampling method will be used.\n')
                    __log_morest.close()
                    raise Exception('Will you use sampling method or not?')
                    
            elif i_parameter.split()[0].upper() == 'Sampling_initialization'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['sampling_initialization'] = True
                    # change MoREST_initialization as False
                    self.morest_parameters['morest_initialization'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['sampling_initialization'] = False
                    
            elif i_parameter.split()[0].upper() == 'Sampling_traj_interval'.upper():
                self.sampling_parameters['sampling_traj_interval'] = int(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'Sampling_clean_rotation'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['sampling_clean_rotation'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['sampling_clean_rotation'] = False
                    
            elif i_parameter.split()[0].upper() == 'Sampling_clean_translation'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.sampling_parameters['sampling_clean_translation'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.sampling_parameters['sampling_clean_translation'] = False
                    
            elif i_parameter.split()[0].upper() == 'Sampling_method'.upper():
                self.sampling_parameters['sampling_method'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Sampling_ensemble'.upper():
                self.sampling_parameters['sampling_ensemble'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Many_body_potential'.upper():
                self.sampling_parameters['many_body_potential'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'Input_file'.upper():
                self.sampling_parameters['input_file'] = str(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ML_potential_model'.upper():
                self.sampling_parameters['ml_potential_model'] = str(i_parameter.split()[1])
                    
            elif i_parameter.split()[0].upper() == 'FD_displacement'.upper():
                self.sampling_parameters['fd_displacement'] = float(i_parameter.split()[1])
                
            ########################## Molecular dynamics #########################
                
            elif i_parameter.split()[0].upper() == 'MD_time_step'.upper():
                self.md_parameters['md_time_step'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'MD_simulation_time'.upper():
                self.md_parameters['md_simulation_time'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'MD_temperature'.upper():
                self.md_parameters['md_temperature'] = float(i_parameter.split()[1])
            
            elif i_parameter.split()[0].upper() == 'NVT_SVR_tau'.upper():
                self.md_parameters['nvt_svr_tau'] = float(i_parameter.split()[1])
                    
            ########################## Enhanced sampling ##########################
            elif i_parameter.split()[0].upper() == 'Enhanced_sampling'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.enhanced_sampling_parameters['enhanced_sampling'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.enhanced_sampling_parameters['enhanced_sampling'] = False
                else:
                    __log_morest.write('It is not clear whether the enhanced sampling will be used.\n')
                    __log_morest.close()
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
                
            elif i_parameter.split()[0].upper() == 'ITS_replica_temperature'.upper():
                tmp_temperature = []
                for i in range(self.its_parameters['its_number_of_replica']):
                    tmp_temperature.append(float(i_parameter.split()[i+1]))
                self.its_parameters['its_replica_temperature'] = np.array(tmp_temperature)
                
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
                
            ########################## Wall potential #############################
            elif i_parameter.split()[0].upper() == 'Wall_potential'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['wall_potential'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['wall_potential'] = False
                else:
                    __log_morest.write('It is not clear whether the wall potential will be used.\n')
                    __log_morest.close()
                    raise Exception('Will you use wall potential or not?')
                    
            elif i_parameter.split()[0].upper() == 'Collective_variable'.upper():
                if i_parameter.split()[1].upper() == 'True'.upper():
                    self.wall_potential_parameters['collective_variable'] = True
                elif i_parameter.split()[1].upper() == 'False'.upper():
                    self.wall_potential_parameters['collective_variable'] = False
                else:
                    __log_morest.write('It is not clear whether the collective variable will be used.\n')
                    __log_morest.close()
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
                
        __log_morest.write('\n')
        try:
            for key in self.morest_parameters:
                __log_morest.write(key+' : '+str(self.self.morest_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.sampling_parameters:
                __log_morest.write(key+' : '+str(self.self.sampling_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.md_parameters:
                __log_morest.write(key+' : '+str(self.self.md_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.enhanced_sampling_parameters:
                __log_morest.write(key+' : '+str(self.self.enhanced_sampling_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.its_parameters:
                __log_morest.write(key+' : '+str(self.self.its_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.wall_potential_parameters:
                __log_morest.write(key+' : '+str(self.self.wall_potential_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.plane_wall_parameters:
                __log_morest.write(key+' : '+str(self.self.plane_wall_parameters[key])+'\n')
        except:
            pass
        try:
            for key in self.spherical_wall_parameters:
                __log_morest.write(key+' : '+str(self.self.spherical_wall_parameters[key])+'\n')
        except:
            pass
        __log_morest.write('\n')
            

                
    def get_morest_parameters(self):
        #np.save('MoREST_morest_parameters.npy', self.morest_parameters)
        return self.morest_parameters

    def get_sampling_parameters(self):
        return self.sampling_parameters
    
    def get_md_parameters(self):
        #np.save('MoREST_md_parameters.npy', self.md_parameters)
        self.md_parameters['md_time_step'] = self.md_parameters['md_time_step'] * units.fs
        self.md_parameters['md_simulation_time'] = self.md_parameters['md_simulation_time'] * units.fs
        self.md_parameters['nvt_svr_tau'] = self.md_parameters['nvt_svr_tau'] * units.fs
        return self.md_parameters

    def get_enhanced_sampling_parameters(self):
        #np.save('MoREST_enhanced_sampling_parameters.npy', self.enhanced_sampling_parameters)
        return self.enhanced_sampling_parameters
        
    def get_its_parameters(self, log_morest):
        __log_morest = log_morest
        if self.its_parameters['its_initialization']:
            try:
                os.remove('MoREST_ITS_pk.npy')
                os.remove('MoREST_ITS_nk.npy')
                os.remove('MoREST_ITS_potential_energy.npy')
            except:
                pass
            __log_morest.write('Integrated tempering sampling method is initialized.\n\n')

        if not 'its_replica_temperature' in self.its_parameters:
            if int(self.its_parameters['its_replica_arrange']) == -1:
                replica_temperature = np.linspace(self.its_parameters['its_lower_bound_temperature'],\
                                                  self.its_parameters['its_upper_bound_temperature'],\
                                                  num=self.its_parameters['its_number_of_replica'],\
                                                  endpoint=True)
            elif int(self.its_parameters['its_replica_arrange']) == 0:
                replica_temperature = np.geomspace(self.its_parameters['its_lower_bound_temperature'],\
                                                   self.its_parameters['its_upper_bound_temperature'],\
                                                   num=self.its_parameters['its_number_of_replica'],\
                                                   endpoint=True)
            else:
                __log_morest.write('No ITS_replica_arrange type was matched.\n')
                __log_morest.close()
                raise Exception('No ITS_replica_arrange type was matched.')
            self.its_parameters['its_replica_temperature'] = replica_temperature
        
        self.its_parameters['its_replica_beta'] = 1/(self.its_parameters['its_replica_temperature'] *\
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
        np.save('MoREST_ITS_parameters.npy',self.its_parameters)
        return self.its_parameters
    
    def get_wall_potential_parameters(self):
        np.save('MoREST_wall_potential_parameters.npy', self.wall_potential_parameters)
        return self.wall_potential_parameters
    
    def get_plane_wall_parameters(self):
        np.save('MoREST_plane_wall_parameters.npy', self.plane_wall_parameters)
        return self.plane_wall_parameters

    def get_spherical_wall_parameters(self):
        np.save('MoREST_spherical_wall_parameters.npy', self.spherical_wall_parameters)
        return self.spherical_wall_parameters
    