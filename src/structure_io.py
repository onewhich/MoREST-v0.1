import numpy as np
import ase.io

def read_xyz_file(xyz_file):
    '''
    Read extxyz structure file, MoREST.str or MoREST.str_new file and return a system (ase.Atoms object)
    '''
    system = ase.io.read(xyz_file, format='extxyz')
    
    return system

def write_xyz_file(xyz_file, system):
    '''
    Write system (ase.Atoms object) to a extxyz format file, MoREST.str_new
    '''
    ase.io.write(xyz_file, system, format='extxyz')
    
def read_xyz_traj(traj_file):
    '''
    Read xyz format trajectory file, MoREST.xyz_traj, and return a system_list (list of ase.Atoms objects)
    '''
    system_list = ase.io.read(traj_file, index=':', format='extxyz')
    
    return system_list
    
def write_xyz_traj(traj_file, system_new):
    '''
    Write the list of system (ase.Atoms object) to a extxyz format file, MoREST.xyz_traj
    '''
    ase.io.write(traj_file, system_new, format='extxyz', append=True)