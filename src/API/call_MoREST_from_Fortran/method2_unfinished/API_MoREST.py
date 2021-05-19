import sys
sys.path.append('../../enhanced_sampling')
import MoREST

def call_MoREST_ITS(if_initial, simulation_temperature, potential_energy,\
                    current_md_step, md_force):
    simulation_maxsteps = 9999 # this value is not used in ITS
    time_step = 0.001 # this value is not used in ITS
    return MoREST.enhanced_sampling('ITS', if_initial, simulation_temperature, simulation_maxsteps, time_step,\
                      potential_energy, current_md_step, md_force, parameter_file='MoREST.in')
