import numpy as np
import periodictable
import units

def read_xyz_file(xyz_file):
    '''
    Read xyz structure file or MoREST.str, MoREST.str_new file and return structure dict{}
    '''
    xyz = open(xyz_file).readlines()
    
    structure = {}
    structure['n_atoms'] = int(xyz[0].split()[0])
    try:
        structure['current_step'] = int(xyz[1].split()[0])
        structure['total_energy'] = float(xyz[1].split()[1])
    except:
        pass
    
    element_list = []
    mass_list = []
    str_x_list = []
    str_y_list = []
    str_z_list = []
    str_vx_list = []
    str_vy_list = []
    str_vz_list = []
    str_fx_list = []
    str_fy_list = []
    str_fz_list = []
    for i_atom in range(structure['n_atoms']):
        element_list.append(str(xyz[i_atom+2].split()[0]))
        str_x_list.append(float(xyz[i_atom+2].split()[1]))
        str_y_list.append(float(xyz[i_atom+2].split()[2]))
        str_z_list.append(float(xyz[i_atom+2].split()[3]))
        try:
            str_vx_list.append(float(xyz[i_atom+2].split()[4]))
            str_vy_list.append(float(xyz[i_atom+2].split()[5]))
            str_vz_list.append(float(xyz[i_atom+2].split()[6]))
        except:
            str_vx_list.append(0.0)
            str_vy_list.append(0.0)
            str_vz_list.append(0.0)
            pass
            
        try:
            str_fx_list.append(float(xyz[i_atom+2].split()[7]))
            str_fy_list.append(float(xyz[i_atom+2].split()[8]))
            str_fz_list.append(float(xyz[i_atom+2].split()[9]))
        except:
            pass
            
    structure['elements'] = element_list
    for i_ele in element_list:
        mass_list.append(periodictable.elements.symbol(i_ele).mass)
    structure['masses'] = np.array(mass_list) * units.Dalton
    
    str_x_list = np.array(str_x_list)
    str_y_list = np.array(str_y_list)
    str_z_list = np.array(str_z_list)
    structure['coordinates'] = np.array([str_x_list, str_y_list, str_z_list]).T
    
    str_vx_list = np.array(str_vx_list)
    str_vy_list = np.array(str_vy_list)
    str_vz_list = np.array(str_vz_list)
    structure['velocities'] = np.array([str_vx_list, str_vy_list, str_vz_list]).T
    
    acceleration_list = []
    if len(str_fx_list) > 0:
        str_fx_list = np.array(str_fx_list)
        str_fy_list = np.array(str_fy_list)
        str_fz_list = np.array(str_fz_list)
        structure['forces'] = np.array([str_fx_list, str_fy_list, str_fz_list]).T
        structure['accelerations'] = np.array([structure['forces'][i_atom]/structure['masses'][i_atom] \
                                          for i_atom in range(len(structure['forces']))])
    
    return structure

def write_xyz_file(xyz_file, structure_dict):
    '''
    Write structure dict{} into a xyz format file or MoREST.str_new, MoREST.traj
    '''
    xyz_file.write(str(structure_dict['n_atoms'])+'\n')
    xyz_file.write(str(structure_dict['current_step'])+'    '+str(structure_dict['total_energy'])+'\n')
    for i_atom in range(structure_dict['n_atoms']):
        xyz_file.write(structure_dict['elements'][i_atom]+'    ')
        for i in range(3):
            xyz_file.write(str(structure_dict['coordinates'][i_atom][i])+'    ')
        for i in range(3):
            xyz_file.write(str(structure_dict['velocities'][i_atom][i])+'    ')
        for i in range(3):
            xyz_file.write(str(structure_dict['forces'][i_atom][i])+'    ')
        xyz_file.write('\n')
        