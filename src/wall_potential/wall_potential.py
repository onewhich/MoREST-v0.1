import sys, os
import numpy as np
#sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'../read_parameters'))
#import read_parameters
sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0],'../wall_potential'))
import Plane_wall

def wall_potential(wall_type, general_coordinate,\
                    parameter_file='MoREST.in'):
    '''
    wall_potential will read the positions of atoms and then add forces of the potential on the atoms.
    INPUT:
    wall_type:                The type of wall potential, e.g. plane_opaque_wall, plane_translucent_wall, sphere, point, line
    general_coordinate:       The positions vector of atoms in the system.
    parameter_file:           The path and name of the parameter file.
    OUTPUT:
    wall_force: the forces of the wall potential on the atoms
    '''
    
    log_morest = open('MoREST.log','a')
    wall_potential_parameters = np.load('MoREST_wall_potential_parameters.npy',allow_pickle=True).item()
    
    if not wall_potential_parameters['collective_variable']:
        if wall_potential_parameters['wall_type'] in ['Plane_opaque_wall', 'plane_opaque_wall']:
            plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
            log_morest.write('The defination of the plane opaque wall: Point in plane, Normal vector\n')
            log_morest.write(plane_wall_parameters['Plane_wall_point'],plane_wall_parameters['Plane_wall_normal_vector'])
            log_morest.write('\n')
            log_morest.write('The plane opaque wall potential and force on atoms: XYZ coordinate, Potential, Forces\n')
            wall_force = []
            for i_coordinate in general_coordinate:
                i_wall_force, i_wall_potential = Plane_wall.plane_opaque_wall().get_opaque_wall_force_potential(i_coordinate)
                wall_force.append(i_wall_force)
                log_morest.write(i_coordinate, i_wall_potential, i_wall_force)
                log_morest.write('\n')
            return np.array(wall_force)
        
        if wall_potential_parameters['wall_type'] in ['Plane_translucent_wall', 'plane_translucent_wall']:
            plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
            log_morest.write('The defination of the plane translucent wall: Point in plane, Normal vector\n')
            log_morest.write(plane_wall_parameters['Plane_wall_point'],plane_wall_parameters['Plane_wall_normal_vector'])
            log_morest.write('\n')
            log_morest.write('The plane translucent wall potential and force on atoms: XYZ coordinate, Potential, Forces\n')
            wall_force = []
            for i_coordinate in general_coordinate:
                i_wall_force, i_wall_potential = Plane_wall.plane_translucent_wall().get_translucent_wall_force_potential(i_coordinate)
                wall_force.append(i_wall_force)
                log_morest.write(i_coordinate, i_wall_potential, i_wall_force)
                log_morest.write('\n')
            return np.array(wall_force)
    
        else:
            log_morest.write('No wall type was matched.\n')
            log_morest.close()
            return np.array([0])
    
    else:
        general_coordinate = CV_to_xyz(general_coordinate) # TODO conversion function is not exist.


