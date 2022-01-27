import numpy as np

def opaque_potential(a, c, vec_gc_b, norm_gc_b):
    prefactor = a/(c**2)
    wall_force = prefactor * ((c**2)/(norm_gc_b**2)-1) * vec_gc_b/norm_gc_b
    wall_potential = prefactor * (1/norm_gc_b) * (norm_gc_b - c)**2
    return wall_force, wall_potential

def translucent__potential(a, c, vec_gc_b, norm_gc_b):
    prefactor = a/(c**4)
    wall_force = 6 * prefactor * (c**2 * norm_gc_b - c * norm_gc_b**2) * vec_gc_b/norm_gc_b
    wall_potential = prefactor * (2*c * norm_gc_b + c**2) * (norm_gc_b -c )**2
    return wall_force, wall_potential

#def mirror_potential(a, c):
    
class opaque_wall:
    '''
    Adding a plane or spherical opaque wall with the potential in the system. The plane is defined by a point in the plane and
    a normal vector of the plane. The sphere is defined by a center point and the radius.
    INPUT:
    xyz_coordinate:        The xyz_coordinate is get or conversed from the general_coordinate which can be either cartesian coordinates or 
                               any collective variables. 
    '''
        
    def __init__(self):
        self.wall_potential_parameters = np.load('MoREST_wall_potential_parameters.npy',allow_pickle=True).item()
        self.a = self.wall_potential_parameters['wall_scaling']
        self.c = self.wall_potential_parameters['wall_scope']
        
    def get_plane_opaque_wall_force_potential(self, xyz_coordinate):
        plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
        # calculate the force and potential on an atom with xyz_coordinate
        vec_gc_b = np.dot((xyz_coordinate - plane_wall_parameters['plane_wall_point']), plane_wall_parameters['plane_wall_normal_vector']) \
                    * plane_wall_parameters['plane_wall_normal_vector']
        norm_gc_b = np.linalg.norm(vec_gc_b)
        #print(vec_gc_b,norm_gc_b)
        if norm_gc_b > self.c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            return opaque_potential(self.a, self.c, vec_gc_b, norm_gc_b)
        
    def get_spherical_opaque_wall_force_potential(self, xyz_coordinate):
        spherical_wall_parameters = np.load('MoREST_spherical_wall_parameters.npy',allow_pickle=True).item()
        # calculate the force and potential on an atom with xyz_coordinate
        vec_direction = spherical_wall_parameters['spherical_wall_center'] - xyz_coordinate
        norm_direction = np.linalg.norm(vec_direction)
        vec_gc_b = vec_direction / norm_direction * (spherical_wall_parameters['spherical_wall_radius'] - norm_direction)
        norm_gc_b = np.linalg.norm(vec_gc_b)
        #print(vec_gc_b,norm_gc_b)
        if norm_gc_b > self.c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            return opaque_potential(self.a, self.c, vec_gc_b, norm_gc_b)
    
    
class translucent_wall:
    '''
    Adding a plane or spherical translucent wall with the potential in the system. The plane is defined by a point on the plane and
    a normal vector of the plane. The sphere is defined by a center point and the radius.
    INPUT:
    xyz_coordinate:        The xyz_coordinate is get or conversed from the general_coordinate which can be either cartesian coordinates or 
                               any collective variables. 
    '''
        
    def __init__(self):
        wall_potential_parameters = np.load('MoREST_wall_potential_parameters.npy',allow_pickle=True).item()
        self.a = wall_potential_parameters['wall_scaling']
        self.c = wall_potential_parameters['wall_scope']
        
    def get_plane_translucent_wall_force_potential(self, xyz_coordinate):
        plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
        # calculate the force and potential on an atom with xyz_coordinate
        vec_gc_b = np.dot((xyz_coordinate - plane_wall_parameters['plane_wall_point']), plane_wall_parameters['plane_wall_normal_vector']) \
                    * plane_wall_parameters['plane_wall_normal_vector']
        norm_gc_b = np.linalg.norm(vec_gc_b)
        if norm_gc_b > self.c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            return translucent__potential(self.a, self.c, vec_gc_b, norm_gc_b)
        
    def get_spherical_translucent_wall_force_potential(self, xyz_coordinate):
        spherical_wall_parameters = np.load('MoREST_spherical_wall_parameters.npy',allow_pickle=True).item()
        # calculate the force and potential on an atom with xyz_coordinate
        vec_direction = spherical_wall_parameters['spherical_wall_center'] - xyz_coordinate
        norm_direction = np.linalg.norm(vec_direction)
        vec_gc_b = vec_direction / norm_direction * (spherical_wall_parameters['spherical_wall_radius'] - norm_direction)
        norm_gc_b = np.linalg.norm(vec_gc_b)
        #print(vec_gc_b,norm_gc_b)
        if norm_gc_b > self.c:
            return np.zeros(np.shape(xyz_coordinate)), 0.
        else:
            return translucent_potential(self.a, self.c, vec_gc_b, norm_gc_b)
        