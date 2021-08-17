import os, sys
import numpy as np
#sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'../read_parameters'))
from read_parameters import read_parameters
#sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'../enhanced_sampling'))
from enhanced_sampling import enhanced_sampling
#sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'../wall_potential'))
from wall_potential import wall_potential


class morest:
    '''
    The Molecular Reaction Simulation Toolkits module.
    '''

    def __init__(self):
        log_morest = open('MoREST.log','w')
        MoREST_parameters = read_parameters(log_morest)

        enhanced_sampling_parameters = MoREST_parameters.get_enhanced_sampling_parameters()
        for key in enhanced_sampling_parameters:
            print(key+' : '+str(enhanced_sampling_parameters[key]))
        if enhanced_sampling_parameters['enhanced_sampling']:
            its_parameters = MoREST_parameters.get_its_parameters()
            for key in its_parameters:
                print(key+' : '+str(its_parameters[key]))

        wall_potential_parameters = MoREST_parameters.get_wall_potential_parameters()
        for key in wall_potential_parameters:
            print(key+' : '+str(wall_potential_parameters[key]))
        if wall_potential_parameters['wall_potential']:
            plane_wall_parameters = MoREST_parameters.get_plane_wall_parameters()
            for key in plane_wall_parameters:
                print(key+' : '+str(plane_wall_parameters[key]))

    
    
   def bias_sampling():

