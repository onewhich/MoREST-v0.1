import numpy as np
import API_MoREST

simulation_temperature = 798 # K
current_md_step = 0

for _ in range(10000):
    if _ == 0:
        if_initial = True
    else:
        if_initial = False
        
    potential_energy = np.random.random_sample()
    md_force = np.random.rand(2,3)
    current_md_step += 1
    
    bias_force, current_md_step = API_MoREST.call_MoREST_ITS(\
                                  if_initial, simulation_temperature, potential_energy,\
                                  current_md_step, md_force)
    
