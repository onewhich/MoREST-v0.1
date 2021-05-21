import sys
sys.path.append('../../enhanced_sampling/')
from api_morest import ffi
import MoREST
import numpy as np
import copy

# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        #print('ctype : ', ctype )
        #print('dtype : ', dtype )
        ctype2dtype[ctype] = np.dtype(dtype)

#ctype2dtype['int'] = np.dtype('int')

# Floating point types
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')

#print(ctype2dtype)

def get_value(ffi, ptr):
    type_value = ffi.getctype(ffi.typeof(ptr).item)
    value = np.frombuffer(ffi.buffer(ptr, ffi.sizeof(type_value)), ctype2dtype[type_value])[0]
    return value

def get_md_force(ffi, ptr, ptr_shape):
    length_md_force_shape = 2
    type_md_force_shape = ffi.getctype(ffi.typeof(ptr_shape).item)
    md_force_shape = np.frombuffer(ffi.buffer(ptr_shape,\
                     length_md_force_shape * ffi.sizeof(type_md_force_shape)),\
                     ctype2dtype[type_md_force_shape])

    length_md_force = np.prod(md_force_shape)
    type_md_force = ffi.getctype(ffi.typeof(ptr).item)
    md_force = np.frombuffer(ffi.buffer(ptr, length_md_force * ffi.sizeof(type_md_force)),\
                     ctype2dtype[type_md_force]).reshape(-1,3)

    return md_force

@ffi.def_extern()
def call_morest_its(ptr_if_initial, ptr_simulation_temperature, ptr_potential_energy,\
                    ptr_current_md_step, ptr_md_force, ptr_md_force_shape):

    simulation_maxsteps = 9999 # This value is not used in ITS
    time_step = 0.001 # This value is not used in ITS
    if_initial = get_value(ffi, ptr_if_initial)
    simulation_temperature = get_value(ffi, ptr_simulation_temperature)
    potential_energy = get_value(ffi, ptr_potential_energy)
    current_md_step = get_value(ffi, ptr_current_md_step)
    md_force = get_md_force(ffi, ptr_md_force, ptr_md_force_shape)

#    print(if_initial, simulation_temperature, potential_energy, current_md_step)
#    print(md_force)
#    print(id(current_md_step))

    bias_force = MoREST.enhanced_sampling('its', if_initial,\
                  simulation_temperature, simulation_maxsteps,\
                  time_step, potential_energy, current_md_step, md_force)

#    for i in range(len(bias_force)):
#        for j in range(len(bias_force[i])):
#            md_force[i,j] = bias_force[i,j]
    np.put(md_force, range(len(bias_force.flatten())), bias_force)

#    with open('zz.txt','a') as zz:
#        zz.write(str(current_md_step)+'\n')
#    print(id(current_md_step))
#    print(current_md_step)

#    print(md_force)
