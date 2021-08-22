import os,sys
import numpy as np
import scipy.constants
import json
from json import JSONEncoder
import json

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
            self.__log_morest.write('Can not open parameter file.\n')
            self.__log_morest.close()
            sys.exit(0)
        
        self.enhanced_sampling_parameters = {}
        self.its_parameters = {}
        self.wall_potential_parameters = {}
        self.plane_wall_parameters = {}
        for i_parameter in __parameters:
            if len(i_parameter.split()) < 2:
                continue
            ########################## Enhanced sampling ##########################
            if i_parameter.split()[0].upper() == 'Enhanced_sampling'.upper():
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
                
            elif i_parameter.split()[0].upper() == 'Plane_wall_scaling'.upper():
                self.plane_wall_parameters['plane_wall_scaling'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'Plane_wall_scope'.upper():
                self.plane_wall_parameters['plane_wall_scope'] = float(i_parameter.split()[1])
                
    def get_enhanced_sampling_parameters(self):
        np.save('MoREST_enhanced_sampling_parameters.npy',self.enhanced_sampling_parameters)
        return self.enhanced_sampling_parameters
        
    def get_its_parameters(self):
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
                self.__log_morest.write('No ITS_replica_arrange type was matched.\n')
                self.__log_morest.close()
                raise Exception('No ITS_replica_arrange type was matched.')
            self.its_parameters['its_replica_temperature'] = replica_temperature
        
        self.its_parameters['its_replica_beta'] = 1/(self.its_parameters['its_replica_temperature'] *\
                                                    scipy.constants.value('Boltzmann constant in eV/K'))
            
        if not 'its_initial_nk' in self.its_parameters:
            self.its_parameters['its_initial_nk'] = np.exp(self.its_parameters['its_replica_beta'])
            self.its_parameters['its_initial_nk'] = self.its_parameters['its_initial_nk'] /\
                                                    np.max(self.its_parameters['its_initial_nk'])
            
        if not 'its_pk0' in self.its_parameters:
            self.its_parameters['its_pk0'] = np.ones((self.its_parameters['its_number_of_replica'])) /\
                                                   self.its_parameters['its_number_of_replica']
            
        #with open('MoREST_ITS_parameters.json','w') as its_json:
        #    json.dump(self.its_parameters,its_json, cls=NumpyArrayEncoder)
        np.save('MoREST_ITS_parameters.npy',self.its_parameters)
        for key in self.its_parameters:
            self.__log_morest.write(key+' : '+str(self.its_parameters[key])+'\n')
        self.__log_morest.write('\n')
        
        if self.its_parameters['its_initialization']:
            if os.path.isfile('MoREST_ITS_pk.npy'):
                os.remove('MoREST_ITS_pk.npy')
            if os.path.isfile('MoREST_ITS_nk.npy'):
                os.remove('MoREST_ITS_nk.npy')
            if os.path.isfile('MoREST_ITS_potential_energy.list'):
                os.remove('MoREST_ITS_potential_energy.list')
            self.__log_morest.write('Start to initialize integrated tempering sampling method.\n\n')
            
        return self.its_parameters
    
    def get_wall_potential_parameters(self):
        np.save('MoREST_wall_potential_parameters.npy', self.wall_potential_parameters)
        return self.wall_potential_parameters
    
    def get_plane_wall_parameters(self):
        '''
        if self.wall_potential_parameters['wall_type'] in ['Plane_opaque_wall', 'plane_opaque_wall']:
            self.__log_morest.write('The defination of the plane opaque wall: Point in plane, Normal vector\n')
            self.__log_morest.write(str(self.plane_wall_parameters['plane_wall_point']) + \
                               ' '+str(self.plane_wall_parameters['plane_wall_normal_vector']))
            self.__log_morest.write('\n')
            self.__log_morest.write('Plane_wall_scaling : '+str(self.plane_wall_parameters['plane_wall_scaling'])+'\n')
            self.__log_morest.write('Plane_wall_scope : '+str(self.plane_wall_parameters['plane_wall_scope'])+'\n')
            self.__log_morest.write('\n')
        if self.wall_potential_parameters['wall_type'] in ['Plane_translucent_wall', 'plane_translucent_wall']:
            self.__log_morest.write('The defination of the plane translucent wall: Point in plane, Normal vector\n')
            self.__log_morest.write(str(self.plane_wall_parameters['plane_wall_point']) + \
                               ' ' + str(self.plane_wall_parameters['plane_wall_normal_vector']))
            self.__log_morest.write('\n')
            self.__log_morest.write('Plane_wall_scaling : '+str(self.plane_wall_parameters['plane_wall_scaling'])+'\n')
            self.__log_morest.write('Plane_wall_scope : '+str(self.plane_wall_parameters['plane_wall_scope'])+'\n')
            self.__log_morest.write('\n')
        '''
        for key in self.plane_wall_parameters:
            self.__log_morest.write(key+' : '+str(self.plane_wall_parameters[key])+'\n')
        self.__log_morest.write('\n')
        np.save('MoREST_plane_wall_parameters.npy', self.plane_wall_parameters)
        return self.plane_wall_parameters
