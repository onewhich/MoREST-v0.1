import numpy as np
import ase.io
from ase.units import kB

def read_xyz_file(xyz_file):
    '''
    Read extxyz structure file and return a system (ase.Atoms object)
    '''
    system = ase.io.read(xyz_file, format='extxyz')
    
    return system

def write_xyz_file(xyz_file, system):
    '''
    Write system (ase.Atoms object) to a extxyz format file
    '''
    ase.io.write(xyz_file, system, format='extxyz')
    
def read_xyz_traj(traj_file):
    '''
    Read xyz format trajectory file, and return a system_list (list of ase.Atoms objects)
    '''
    system_list = ase.io.read(traj_file, index=':', format='extxyz')
    
    return system_list
    
def write_xyz_traj(traj_file, system_new):
    '''
    Write the list of system (ase.Atoms object) to a extxyz format file
    '''
    ase.io.write(traj_file, system_new, format='extxyz', append=True)

def output_energy_from_xyz_traj(logfilename, traj_file):
    '''
    Output the potential energy, the kinetic energy, the temperature of the kinetic energy, the total energy
    '''
    logfile = open(logfilename,'w')
    traj = read_xyz_traj(traj_file)
    logfile.write('# RPMD step, Potential energy (eV), Kinetic energy (eV), Instant temperature (K), Total energy (eV)\n')  
    n_atom = traj[0].get_global_number_of_atoms()
    for i,i_sys in enumerate(traj):
        Ep = i_sys.get_potential_energy()
        Ek = i_sys.get_kinetic_energy()
        T = 2/3 * Ek/kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
        Et = Ek + Ep
        logfile.write(str(i)+'    '+str(Ep)+'    '+str(Ek)+'    '+str(T)+'    '+str(Et)+'\n')
