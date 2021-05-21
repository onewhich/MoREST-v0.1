import sys
sys.path.append('../../enhanced_sampling/')
import numpy as np
import MoREST

simulation_temperature = 798 # K
simulation_maxsteps = 10000
time_step = 0.0001 # ps
#potential_energy = -20 # eV
#current_md_step = 1000
#md_force = np.array([[1.,2.,3.],[1.,3.,2.],[3.,1.,2.],[3.,2.,1.],[1.,2.,3.],[1.,3.,2.],[3.,1.,2.],[3.,2.,1.]])


current_md_step = 0
for _ in range(1000):
    if _ == 0:
        if_initial = True
    else:
        if_initial = False
        
    potential_energy = np.random.random_sample()
    md_force = np.random.rand(2,3)
    current_md_step += 1
    
    bias_force, current_md_step = MoREST.enhanced_sampling('its', if_initial,\
                  simulation_temperature, simulation_maxsteps,\
                  time_step, potential_energy, current_md_step, md_force)
    
