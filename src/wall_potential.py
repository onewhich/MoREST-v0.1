#from distutils.sysconfig import get_config_h_filename
import numpy as np
    
class repulsive_wall:
    '''
    Adding multiple walls with shape and potential in the system.
    The plane is defined by a point in the plane and a normal vector of the plane.
    The sphere is defined by a center point and the radius.
    The dot is defined by the coordinate.
    INPUT:
    xyz_coordinate:        The xyz_coordinate is get or conversed from the general_coordinate which can be either cartesian coordinates or 
                               any collective variables. 
    '''
        
    def __init__(self, wall_potential_parameters):
        #self.wall_potential_parameters = np.load('MoREST_wall_potential_parameters.npy',allow_pickle=True).item()
        self.wall_potential_parameters = wall_potential_parameters
        self.a = self.wall_potential_parameters['wall_scaling']
        self.c = self.wall_potential_parameters['wall_scope']

    def get_repulsive_wall_force(self, general_coordinate):
        wall_force = np.zeros(np.shape(general_coordinate))
        for i in range(self.wall_potential_parameters['wall_number']):
            if not self.wall_potential_parameters['wall_collective_variable'][i]:
                self.xyz_coordinate = general_coordinate
            else:
                self.xyz_coordinate = CV_to_XYZ(general_coordinate)    # TODO CV_to_XYZ does not exist
            vec_gc_b, norm_gc_b = self.get_gc_b(i)
            tmp_force, tmp_potential = self.get_potential(i, self.a[i], self.c[i], vec_gc_b, norm_gc_b)
            wall_force += tmp_force
        return wall_force

    def get_repulsive_wall_force_potential(self, general_coordinate):
        wall_force = np.zeros(np.shape(general_coordinate))
        wall_potential = 0
        for i in range(self.wall_potential_parameters['wall_number']):
            if not self.wall_potential_parameters['wall_collective_variable'][i]:
                self.xyz_coordinate = general_coordinate
            else:
                self.xyz_coordinate = CV_to_XYZ(general_coordinate)    # TODO CV_to_XYZ does not exist
            vec_gc_b, norm_gc_b = self.get_gc_b(i)
            tmp_force, tmp_potential = self.get_potential(i, self.a[i], self.c[i], vec_gc_b, norm_gc_b)
            wall_force += tmp_force
            wall_potential += tmp_potential
        return wall_force, wall_potential

    def get_gc_b(self, i_wall):
        if self.wall_potential_parameters['wall_shape'][i_wall] == 'planar':
            return self.get_planar_wall_gc_b(i_wall)
        elif self.wall_potential_parameters['wall_shape'][i_wall] == 'spherical':
            return self.get_spherical_wall_gc_b(i_wall)
        elif self.wall_potential_parameters['wall_shape'][i_wall] == 'dot':
            return self.get_dot_wall_gc_b(i_wall)

    def get_planar_wall_gc_b(self, i_wall):
        vec_gc_b = np.dot((self.xyz_coordinate - self.wall_potential_parameters['wall_shape_parameters'][i_wall]['planar_wall_point']), \
                self.wall_potential_parameters['wall_shape_parameters'][i_wall]['planar_wall_normal_vector']) * \
                    self.wall_potential_parameters['wall_shape_parameters'][i_wall]['planar_wall_normal_vector']
        norm_gc_b = np.linalg.norm(vec_gc_b)
        return vec_gc_b, norm_gc_b

    def get_spherical_wall_gc_b(self, i_wall):
        vec_direction = self.wall_potential_parameters['wall_shape_parameters'][i_wall]['spherical_wall_center'] - self.xyz_coordinate
        norm_direction = np.linalg.norm(vec_direction)
        vec_gc_b = vec_direction / norm_direction * \
                   (self.wall_potential_parameters['wall_shape_parameters'][i_wall]['spherical_wall_radius'] - norm_direction)
        norm_gc_b = np.linalg.norm(vec_gc_b)
        return vec_gc_b, norm_gc_b

    def get_dot_wall_gc_b(self, i_wall):
        vec_gc_b = self.xyz_coordinate - self.wall_potential_parameters['wall_shape_parameters'][i_wall]['dot_wall_position']
        norm_gc_b = np.linalg.norm(vec_gc_b)
        return vec_gc_b, norm_gc_b

    def get_potential(self, i_wall, a, c, vec_gc_b, norm_gc_b):
        if self.wall_potential_parameters['wall_type'][i_wall] == 'opaque_wall':
            return self.opaque_potential(a, c, vec_gc_b, norm_gc_b)
        elif self.wall_potential_parameters['wall_type'][i_wall] == 'translucent_wall':
            return self.translucent_potential(a, c, vec_gc_b, norm_gc_b)
        elif self.wall_potential_parameters['wall_type'][i_wall] == 'power_wall':
            return self.power_potential(a, c, vec_gc_b, norm_gc_b, i_wall)

    def opaque_potential(self, a, c, vec_gc_b, norm_gc_b):
        prefactor = a/(c**2)
        wall_force = prefactor * ((c**2)/(norm_gc_b**2)-1) * vec_gc_b/norm_gc_b
        wall_potential = prefactor * (1/norm_gc_b) * (norm_gc_b - c)**2
        if norm_gc_b > c:
            return np.zeros(np.shape(vec_gc_b)), 0.
        else:
            return wall_force, wall_potential

    def translucent_potential(self, a, c, vec_gc_b, norm_gc_b):
        prefactor = a/(c**4)
        wall_force = 6 * prefactor * (c**2 * norm_gc_b - c * norm_gc_b**2) * vec_gc_b/norm_gc_b
        wall_potential = prefactor * (2*c * norm_gc_b + c**2) * (norm_gc_b -c )**2
        if norm_gc_b > c:
            return np.zeros(np.shape(vec_gc_b)), 0.
        else:
            return wall_force, wall_potential

    def power_potential(self, a, c, vec_gc_b, norm_gc_b, i_wall):
        # c should be larger than 1
        wall_force = -1 * a * c * norm_gc_b**(c-1) * vec_gc_b/norm_gc_b
        wall_potential = a * norm_gc_b**c
        if self.wall_potential_parameters['wall_shape'][i_wall] == 'planar':
            if np.dot(norm_gc_b,self.wall_potential_parameters['wall_shape_parameters'][i_wall]['planar_wall_normal_vector']) * \
                self.wall_potential_parameters['power_wall_direction'][i_wall] <= 0:
                return np.zeros(np.shape(vec_gc_b)), 0.
            else:
                return wall_force, wall_potential
        elif self.wall_potential_parameters['wall_shape'][i_wall] == 'spherical':
            if (np.linalg.norm(self.xyz_coordinate - self.wall_potential_parameters['wall_shape_parameters'][i_wall]['spherical_wall_center']) \
                - self.wall_potential_parameters['wall_shape_parameters'][i_wall]['spherical_wall_radius']) \
                    * self.wall_potential_parameters['power_wall_direction'][i_wall] <= 0:
                return np.zeros(np.shape(vec_gc_b)), 0.
            else:
                return wall_force, wall_potential
        elif self.wall_potential_parameters['wall_shape'][i_wall] == 'dot':
            if self.wall_potential_parameters['power_wall_direction'][i_wall] <= 0:
                return np.zeros(np.shape(vec_gc_b)), 0.
            else:
                return wall_force, wall_potential
        
    def update_wall_parameters(self, wall_potential_parameters):
        self.__init__(self, wall_potential_parameters)

    #def mirror_potential(a, c):
    
