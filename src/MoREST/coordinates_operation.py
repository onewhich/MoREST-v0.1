import numpy as np

def reset_geometric_center(system, center=[0., 0., 0.]):
    '''
    set the geometric center to specified center
    '''
    coordinates = system.get_positions()
    n_atom = system.get_global_number_of_atoms()
    geometric_center = np.sum(coordinates, axis=0)/n_atom
    system.set_positions(coordinates - geometric_center + center)

def reset_mass_center(system, center=[0., 0., 0.]):
    '''
    set the mass center to specified center
    '''
    center = np.array(center)
    coordinates = system.get_positions()
    masses = system.get_masses()[:,np.newaxis]
    mass_center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    system.set_positions(coordinates - mass_center + center)

def rotate_system_at_center(system, theta, unit_normal_vector, center=[0., 0., 0.]):
    '''
    theta: angles, in radian
    center: 1-D list of 3 numbers (define the position of the center), or 'geometry' (calculate geometric center), or 'mass' (calculate mass' center)
    '''
    # (r x r_new) / (|r| * |r_new|) = sin(theta) * unit_normal_vector, where |r| == |r_new|
    # r x r_new = sin(theta) * unit_normal_vector * |r|^2

    unit_normal_vector = np.array(unit_normal_vector, dtype=float)
    norm_n = np.linalg.norm(unit_normal_vector)
    if norm_n == 0:
        raise ValueError("The unit_normal_vector cannot be a zero vector.")
    unit_normal_vector = unit_normal_vector / norm_n

    if isinstance(center, (list, np.ndarray)):
        center = np.array(center, dtype=float)
    elif center.lower() == 'geometry':
        center = np.sum(coordinates, axis=0)/len(coordinates)
    elif center.lower() == 'mass':
        #masses = system.get_masses()[:,np.newaxis]
        #center = np.sum(masses*coordinates, axis=0)/np.sum(masses)
        center = system.get_center_of_mass()

    coordinates = system.get_positions()
    shifted_coordinates = coordinates - center
    # Apply Rodrigues' Rotation Formula
    # v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    term1 = shifted_coordinates * np.cos(theta)
    term2 = np.cross(unit_normal_vector, shifted_coordinates) * np.sin(theta)
    term3_scalar = np.dot(shifted_coordinates, unit_normal_vector) * (1 - np.cos(theta))
    term3 = unit_normal_vector * term3_scalar[:, np.newaxis]
    rotated_coordinates = term1 + term2 + term3
    
    rotated_coordinates_shifted_back = rotated_coordinates + center
    system_new = system.copy()
    system_new.set_positions(rotated_coordinates_shifted_back)

    return system_new

def rotate_at_center(coordinates, theta, unit_normal_vector, center=[0., 0., 0.]):
    """
    Rotates a set of coordinates around a specified center point
    by an angle theta about an arbitrary axis defined by a unit normal vector.
    Args:
        coordinates (np.ndarray): An N x 3 numpy array of points to rotate.
        theta (float): The angle of rotation in radians.
        unit_normal_vector (np.ndarray): A 3-element numpy array representing the
                                        unit vector of the rotation axis.
        center (np.ndarray, optional): A 3-element numpy array representing the
                                    center of rotation. Defaults to [0,0,0]
                                    if None is provided.
    Returns:
        np.ndarray: The N x 3 numpy array of rotated coordinates.
    """
    center = np.array(center, dtype=float)
    unit_normal_vector = np.array(unit_normal_vector, dtype=float)
    # Ensure the normal vector is a unit vector
    norm_n = np.linalg.norm(unit_normal_vector)
    if norm_n == 0:
        raise ValueError("The unit_normal_vector cannot be a zero vector.")
    unit_normal_vector = unit_normal_vector / norm_n
    # Convert coordinates to numpy array and shift to origin for rotation
    coordinates = np.array(coordinates, dtype=float)
    shifted_coordinates = coordinates - center
    # Apply Rodrigues' Rotation Formula
    # v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    # Component 1: v * cos(theta)
    term1 = shifted_coordinates * np.cos(theta)
    # Component 2: (k x v) * sin(theta)
    # np.cross handles broadcasting when one argument has more dimensions
    term2 = np.cross(unit_normal_vector, shifted_coordinates) * np.sin(theta)
    # Component 3: k * (k . v) * (1 - cos(theta))
    # np.dot(shifted_coordinates, unit_normal_vector) computes the dot product
    # for each row of shifted_coordinates with unit_normal_vector
    term3_scalar = np.dot(shifted_coordinates, unit_normal_vector) * (1 - np.cos(theta))
    term3 = unit_normal_vector * term3_scalar[:, np.newaxis] # Reshape term3_scalar for broadcasting
    # Sum the terms
    rotated_coordinates = term1 + term2 + term3
    # Shift coordinates back from origin
    rotated_coordinates_shifted_back = rotated_coordinates + center
    return rotated_coordinates_shifted_back