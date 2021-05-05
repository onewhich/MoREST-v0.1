import sys
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
    ITS parameters are stored in dictionary named 'its_parameters'.
    '''
    
    def __init__(self, log_morest, parameter_file='MoREST.parameter'):
        self.__log_morest = log_morest
        try:
            self.__parameters = open(parameter_file,'r').readlines()
        except FileNotFoundError:
            self.__log_morest.write('Can not open parameter file.\n')
            self.__log_morest.close()
            sys.exit(0)
        
        self.its_parameters = {}
        for i_parameter in self.__parameters:
            # ITS parameters
            if i_parameter.split()[0].upper() == 'ITS_lower_bound_temperature'.upper():
                self.its_parameters['its_lower_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_upper_bound_temperature'.upper():
                self.its_parameters['its_upper_bound_temperature'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_number_of_replica'.upper():
                self.its_parameters['its_number_of_replica'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_replica_arrange'.upper():
                self.its_parameters['its_replica_arrange'] = float(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_replica_temperature'.upper():
                self.its_parameters['its_replica_temperature'] = np.array(i_parameter.split()[1])
                # TODO: read array labeled by [] including blank space ‘ ’
                
            elif i_parameter.split()[0].upper() == 'ITS_initial_nk'.upper():
                self.its_parameters['its_initial_nk'] = np.array(i_parameter.split()[1])
                # TODO: read array labeled by [] including blank space ‘ ’
                
            elif i_parameter.split()[0].upper() == 'ITS_pk0'.upper():
                self.its_parameters['its_pk0'] = np.array(i_parameter.split()[1])
                # TODO: read array labeled by [] including blank space ‘ ’
                
            elif i_parameter.split()[0].upper() == 'ITS_trial_MD_steps'.upper():
                self.its_parameters['its_trial_MD_steps'] = int(i_parameter.split()[1])
                
            elif i_parameter.split()[0].upper() == 'ITS_delta_pk'.upper():
                self.its_parameters['its_delta_pk'] = float(i_parameter.split()[1])
                
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
            
        if not 'its_initial_nk' in self.its_parameters:
            self.its_parameters['its_initial_nk'] = np.exp(1/(self.its_parameters['its_replica_temperature'] *\
                                                    scipy.constants.value('Boltzmann constant in eV/K')))
            self.its_parameters['its_initial_nk'] = self.its_parameters['its_initial_nk'] /\
                                                    np.min(self.its_parameters['its_initial_nk'])
            
        if not 'its_pk0' in self.its_parameters:
            self.its_parameters['its_pk0'] = np.ones((self.its_parameters['its_number_of_replica'])) /\
                                                   self.its_parameters['its_number_of_replica']
        #with open('MoREST_ITS_parameters.json','w') as its_json:
        #    json.dump(self.its_parameters,its_json, cls=NumpyArrayEncoder)
        np.save('MoREST_ITS_parameters.npy',self.its_parameters)
        return self.its_parameters