import numpy as np

class plane_opaque_wall:
    '''
    Adding a plane opaque wall with the potential in the system. The plane is defined by a point in the plane and
    a normal vector of the plane.
    INPUT:
    xyz_coordinate:        The xyz_coordinate is get or conversed from the general_coordinate which can be either cartesian coordinates or 
                               any collective variables. 
    '''
        
    def __init__(self):
        self.plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
        
    def get_opaque_wall_force_potential(self, xyz_coordinate):
        # calculate the force and potential on an atom with xyz_coordinate
        a = self.plane_wall_parameters['plane_wall_scaling']
        c = self.plane_wall_parameters['plane_wall_scope']
        vec_gc_b = np.dot((xyz_coordinate - self.plane_wall_parameters['plane_wall_point']),self.plane_wall_parameters['plane_wall_normal_vector'])
        norm_gc_b = np.linalg.norm(vec_gc_b)
        if norm_gc_b > c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            prefactor = a/(c^2)
            wall_force = prefactor * ((c^2)/(norm_gc_b^2)-1) * vec_gc_b/norm_gc_b
            wall_potential = prefactor * (1/norm_gc_b) * (norm_gc_b - c)^2
            return wall_force, wall_potential
    
    
class plane_translucent_wall:
    '''
    Adding a plane translucent wall with the potential in the system. The plane is defined by a point on the plane and
    a normal vector of the plane.
    INPUT:
    xyz_coordinate:        The xyz_coordinate is get or conversed from the general_coordinate which can be either cartesian coordinates or 
                               any collective variables. 
    '''
        
    def __init__(self):
        self.plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
        
    def get_translucent_wall_force_potential(self, xyz_coordinate):
        # calculate the force and potential on an atom with xyz_coordinate
        a = self.plane_wall_parameters['plane_wall_scaling']
        c = self.plane_wall_parameters['plane_wall_scope']
        vec_gc_b = np.dot((xyz_coordinate - self.plane_wall_parameters['plane_wall_point']),self.plane_wall_parameters['plane_wall_normal_vector'])
        norm_gc_b = np.linalg.norm(vec_gc_b)
        if norm_gc_b > c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            prefactor = a/(c^4)
            wall_force = 6 * prefactor * (c^2 * norm_gc_b - c * norm_gc_b^2) * vec_gc_b/norm_gc_b
            wall_potential = prefactor * (2*c * norm_gc_b + c^2) * (norm_gc_b -c )^2
            return wall_force, wall_potential
        