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

def get_kinetic_energies(system):
    '''
    Ek = E_translation + E_rotation + E_vibration
    '''
    masses = system.get_masses()
    translation_velocities, rotation_velocities, vibration_velocities = get_kinetic_velocities(system)

    E_translation = np.sum(0.5 * masses * np.linalg.norm(translation_velocities)**2)

    E_rotation = np.sum(0.5 * masses * np.linalg.norm(rotation_velocities, axis=1)**2)

    E_vibration = np.sum(0.5 * masses * np.linalg.norm(vibration_velocities, axis=1)**2)

    return E_translation, E_rotation, E_vibration
    


def get_translation_velocities(system):
    masses = system.get_masses()[:,np.newaxis]
    velocities = system.get_velocities()
    return np.sum(velocities*masses, axis=0)/np.sum(masses)

def clean_translation(system, preserve_temperature=False):
    temperature = system.get_temperature()
    velocities = system.get_velocities()
    center_of_mass_velocity = get_translation_velocities(system)
    new_velocities = velocities - center_of_mass_velocity
    if preserve_temperature:
        system.set_velocities(new_velocities)
        rescale_T_kinetic(system, temperature)
    else:
        system.set_velocities(new_velocities)

def clean_translation_vm(velocities, masses):
    masses = masses[:,np.newaxis]
    center_of_mass_velocity = np.sum(velocities*masses, axis=0)/np.sum(masses)
    return velocities - center_of_mass_velocity

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
    masses = system.get_masses()[:,np.newaxis]
    if n_atom == 1:
        return velocities
    r_vector = coordinates - center_of_mass
    v_vector = velocities - get_translation_velocities(system)
    # angular momentum
    L_vector = np.sum(np.cross(r_vector, masses * v_vector), axis=0)
    # moment of inertia tensor
    I_tensor = np.zeros((3,3))
    for i in range(n_atom):
        r_i = r_vector[i]
        m_i = masses[i][0]
        I_tensor += m_i * (np.linalg.norm(r_i)**2 * np.identity(3) - np.outer(r_i, r_i))
    if L_vector.any() < 1e-30:
        #omega = np.zeros(3)
        v_tang = np.zeros(3)
    else:
        # angular velocity
        # angular velocity is the same for all atoms
        omega = np.linalg.solve(I_tensor, L_vector)
        # linear velocity
        # v = omega x r
        v_tang = np.cross(omega, r_vector)
    return v_tang

def clean_rotation(system, preserve_temperature=False):
    temperature = system.get_temperature()
    velocities = system.get_velocities()
    v_tang = get_rotation_velocities(system)
    new_velocities = velocities - v_tang
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
    v_vector = clean_translation_vm(velocities, masses)
    masses = masses[:,np.newaxis]
    center_of_mass = np.sum(masses*coordinates, axis=0)/np.sum(masses)
    r_vector = coordinates - center_of_mass
    # angular momentum
    L_vector = np.sum(np.cross(r_vector, masses * v_vector), axis=0)
    # moment of inertia tensor
    I_tensor = np.zeros((3,3))
    for i in range(n_atom):
        r_i = r_vector[i]
        m_i = masses[i][0]
        I_tensor += m_i * (np.linalg.norm(r_i)**2 * np.identity(3) - np.outer(r_i, r_i))
    # angular velocity
    # angular velocity is the same for all atoms
    omega = np.linalg.solve(I_tensor, L_vector)
    # linear velocity
    # v = omega x r
    v_tang = np.cross(omega, r_vector)
    return velocities - v_tang

    
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




