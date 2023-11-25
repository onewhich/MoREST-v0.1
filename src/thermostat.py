import numpy as np
from ase import units

def velocity_rescaling(dT, T_simulation, Ek, n_atom, velocities):
    lower_T = T_simulation - dT
    upper_T = T_simulation + dT
    Ti = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    if Ti > upper_T or Ti < lower_T:
        factor = np.sqrt(T_simulation / Ti)
        new_velocities = factor * velocities

    return new_velocities

def Berendsen_velocity_rescaling(time_step, Ek, n_atom, T_simulation, tau, velocities):
    Ti = 2/3 * Ek/units.kB /n_atom   # Ek = 1/2 m v^2 = 3/2 kB T for each particle
    factor = np.sqrt(1 + time_step/tau * (T_simulation/Ti -1))
    new_velocities = factor * velocities

    return new_velocities

def stochastic_velocity_rescaling(time_step, Ek_t, K_simulation, Nf, tau, velocities):
    '''
    This function implements stochastic velocity rescaling algorithm (Bussi, Donadio and Parrinello, JCP (2007); Bussi, Parrinello, CPC (2008)) to do canonical ensenmble sampling (NVT MD).
    '''
    
    ### degree of freedom
    # Nf = 1                # for Langevin thermostat
    # Nf = 3 * self.n_atom  # for SVR thermostat
    #if self.md_parameters['md_clean_translation']:
    #    Nf = Nf - 3
    #if self.md_parameters['md_clean_rotation']:
    #Nf = Nf - 3
        
    ### Gaussian random number R(t)
    R = np.random.normal(size=Nf)
    R_t = R[0]
    S_Nf_1 = np.sum(R[1:]**2)
    
    ### c = exp(- time_step / tau)
    c = np.exp(-1 * time_step / tau )
    
    ### kinetic energy K
    factor = K_simulation / Ek_t / Nf
    
    ### alpha
    alpha2 = np.abs(c + (1-c)*(S_Nf_1 + R_t**2)*factor + 2*R_t*np.sqrt(c*(1-c)*factor))
    sign = np.sign(R_t + np.sqrt(c/(1-c)/factor))
    alpha = sign * np.sqrt(alpha2)
    
    new_velocities = alpha * velocities

    #d_Ee = -1*((self.K_simulation - K_t)*time_step/tau + 2*np.sqrt(K_t*self.K_simulation/Nf/tau)*R_t)
    d_Ee = Ek_t*(1-alpha2)
    
    return new_velocities, d_Ee, alpha
