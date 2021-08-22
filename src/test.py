from MoREST import morest
import numpy as np

test_morest = morest()

simulation_temperature = 798 # K
simulation_maxsteps = 100000
time_step = 0.0001 # ps
#potential_energy = -20 # eV
#current_md_step = 1000
potential_energy = 0.2
coordinate = np.array([[3.,1.,2.],[3.,2.,1.]])
md_force = np.array([[1.,2.,3.],[1.,3.,2.]])
#md_force = np.array([[1.,2.,3.],[1.,3.,2.],[3.,1.,2.],[3.,2.,1.],[1.,2.,3.],[1.,3.,2.],[3.,1.,2.],[3.,2.,1.]])


current_md_step = 0
for _ in range(100):
    if _ == 0:
        if_initial = True
    else:
        if_initial = False
        
    #potential_energy = np.random.random_sample()
    #md_force = np.random.rand(2,3)
    #coordinate = np.random.rand(2,3)
    current_md_step += 1
    print(coordinate)
    print('----')
    print(md_force)
    print('----')
    bias_force = test_morest.bias_sampling(if_initial, simulation_temperature, simulation_maxsteps,\
                                          time_step, potential_energy, current_md_step, md_force, coordinate)
    print(bias_force)
    print('====')
