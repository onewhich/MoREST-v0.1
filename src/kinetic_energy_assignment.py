import numpy as np
from ase import units

'''
This module is used to assign kinetic energy to a system of particles.
It can be used to assign kinetic energy to a system on rotation and vibration.
'''

def get_kinetic_velocities(system):
    '''
    Ek = E_translation + E_rotation + E_vibration
    '''
    velocities = system.get_velocities()
    translation_velocities = get_translation_velocities(system)
    rotation_velocities = get_rotation_velocities(system)
    vibration_velocities = velocities - translation_velocities - rotation_velocities
    return translation_velocities, rotation_velocities, vibration_velocities

def get_kinetic_temperatures(system):
    '''
    Ek = E_translation + E_rotation + E_vibration
    '''
    masses = system.get_masses()
    translation_velocities, rotation_velocities, vibration_velocities = get_kinetic_velocities(system)

    E_translation = np.sum(0.5 * masses * np.linalg.norm(translation_velocities)**2)
    T_translation = 2/3 * E_translation / system.get_global_number_of_atoms() / units.kB   # Ek = 1/2 m v^2 = 3/2 kB T for each particle

    E_rotation = np.sum(0.5 * masses * np.linalg.norm(rotation_velocities, axis=1)**2)
    T_rotation = 2/3 * E_rotation / system.get_global_number_of_atoms() / units.kB   # Ek = 1/2 m v^2 = 3/2 kB T for each particle

    E_vibration = np.sum(0.5 * masses * np.linalg.norm(vibration_velocities, axis=1)**2)
    T_vibration = 2/3 * E_vibration / system.get_global_number_of_atoms() / units.kB   # Ek = 1/2 m v^2 = 3/2 kB T for each particle

    return T_translation, T_rotation, T_vibration
    


def get_translation_velocities(system):
    velocities = system.get_velocities()
    return np.sum(velocities, axis=0)/len(velocities)

def clean_translation(system, preserve_temperature=False):
    temperature = system.get_temperature()
    velocities = system.get_velocities()
    total_velocity = np.sum(velocities, axis=0)/len(velocities)
    new_velocities = velocities - total_velocity
    if preserve_temperature:
        system.set_velocities(new_velocities)
        rescale_T_kinetic(system, temperature)
    else:
        system.set_velocities(new_velocities)

def clean_translation_v(velocities):
    net_velocity = np.sum(velocities, axis=0)/len(velocities)
    return velocities - net_velocity

def get_rotation_velocities(system):
    '''
    L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
    L : angular momentum
    omega: angular velocity
    I : moment of inertia
    '''
    velocities = system.get_velocities()
    coordinates = system.get_positions()
    n_atom = system.get_global_number_of_atoms()
    center_of_mass = system.get_center_of_mass()
    if n_atom == 1:
        return velocities
    v_vector = velocities
    r_vector = coordinates - center_of_mass
    # r_cross_v : angular velocities
    # omega = (r x v) / |r|^2
    r_cross_v = np.cross(r_vector, v_vector)
    r_2 = np.linalg.norm(r_vector, axis=1)**2
    omega = np.array([r_cross_v[i]/r_2[i] for i in range(n_atom)])
    # Rv = omega/n_atom : system total angular velocity
    rotat_vector = np.sum(omega, axis=0)/n_atom
    v_tang = np.cross(rotat_vector, r_vector)
    return v_tang

def clean_rotation(system, preserve_temperature=False):
    '''
    L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
    L : angular momentum
    omega: angular velocity
    I : moment of inertia
    '''
    temperature = system.get_temperature()
    velocities = system.get_velocities()
    coordinates = system.get_positions()
    n_atom = system.get_global_number_of_atoms()
    center_of_mass = system.get_center_of_mass()
    if n_atom == 1:
        return velocities
    v_vector = velocities
    r_vector = coordinates - center_of_mass
    # r_cross_v : angular velocities
    # omega = (r x v) / |r|^2
    r_cross_v = np.cross(r_vector, v_vector)
    r_2 = np.linalg.norm(r_vector, axis=1)**2
    omega = np.array([r_cross_v[i]/r_2[i] for i in range(n_atom)])
    # Rv = omega/n_atom : system total angular velocity
    rotat_vector = np.sum(omega, axis=0)/n_atom
    v_tang = np.cross(rotat_vector, r_vector)
    new_velocities = v_vector - v_tang
    if preserve_temperature:
        system.set_velocities(new_velocities)
        rescale_T_kinetic(system, temperature)
    else:
        system.set_velocities(new_velocities)

def clean_rotation_vcm(velocities, coordinates, masses):
    '''
    L = r x p = r x (m v) = r x (omega x (m r)) = m r^2 omega = I omega
    L : angular momentum
    omega: angular velocity
    I : moment of inertia
    '''
    n_atom = len(velocities)
    if n_atom == 1:
        return velocities
    masses = masses[:,np.newaxis]
    v_vector = velocities
    #center_of_mass = np.sum([masses[i]*coordinates[i] for i in range(len(masses))], axis=0)/np.sum(masses)
    #center_of_mass = np.sum(masses[:,np.newaxis]*coordinates, axis=0)/np.sum(masses)
    center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    r_vector = coordinates - center_of_mass
    # r_cross_v : angular velocities
    # omega = (r x v) / |r|^2
    r_cross_v = np.cross(r_vector, v_vector)
    r_2 = np.linalg.norm(r_vector, axis=1)**2
    omega = np.array([r_cross_v[i]/r_2[i] for i in range(n_atom)])
    # Rv = omega/n_atom : system total angular velocity
    rotat_vector = np.sum(omega, axis=0)/n_atom
    v_tang = np.cross(rotat_vector, r_vector)
    return v_vector - v_tang

    
def rescale_T_kinetic(system, Tf):
    #n_atom = system.get_global_number_of_atoms()
    #Ek_i = system.get_kinetic_energy()
    #Ti = 2/3 * Ek_i/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    Ti = system.get_temperature()
    velocities = system.get_velocities()
    factor = np.sqrt(Tf / Ti)
    system.set_velocities(factor * velocities)

def rescale_kinetic_temperature(velocities, Ti, Tf):
    factor = np.sqrt(Tf / Ti)
    return factor * velocities




