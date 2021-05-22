import os
import numpy as np
import scipy.constants

class its:
    '''
    The integrated tempering sampling module.
    '''
    
    def __init__(self):
        self.its_parameters = np.load('MoREST_ITS_parameters.npy',allow_pickle=True).item()
        #self.log_its = open('MoREST_ITS.log','w')
        
        
    def its_optimization(self, simulation_temperature, potential_energy, current_md_step, md_force, log_morest):
        #print(current_md_step)
        if current_md_step % self.its_parameters['its_trial_MD_steps'] == 0 :
            if current_md_step != 0 :
                #print('opting')
                #current_md_step = 0
                p_k, n_k = self.__pk_nk()
            
                log_morest.write('Current p_k:    ')
                for i_p in p_k:
                    log_morest.write(str(i_p)+'    ')
                log_morest.write('\n')
                log_morest.write('Current n_k:    ')
                for i_n in n_k:
                    log_morest.write(str(i_n)+'    ')
                log_morest.write('\n\n')
            
                new_nk = n_k * self.its_parameters['its_pk0'] / p_k
                np.savetxt('MoREST_ITS_nk.npy',new_nk)
                bias_force = self.__bias_force(simulation_temperature, potential_energy, md_force)
                os.remove('MoREST_ITS_potential_energy.list')
                return bias_force#, current_md_step
        else:
            #print('not opting')
            with open('MoREST_ITS_potential_energy.list','a') as potential_energy_list:
                potential_energy_list.write(str(potential_energy)+'\n')
            bias_force = self.__bias_force(simulation_temperature, potential_energy, md_force)
            return bias_force#, current_md_step
            
       
    def its_if_converge(self):
        if not os.path.isfile('MoREST_ITS_pk.npy'):
            return False
        p_k = np.loadtxt('MoREST_ITS_pk.npy')
        if abs(np.max(p_k - self.its_parameters['its_pk0'])) < self.its_parameters['its_delta_pk']:
            if os.path.isfile('MoREST_ITS_potential_energy.list'):
                os.remove('MoREST_ITS_potential_energy.list')
            return True
        else:
            return False
        
    def its_sampling(self, simulation_temperature, potential_energy, md_force):
        #print('sampling')
        bias_force = self.__bias_force(simulation_temperature, potential_energy, md_force)
        return bias_force
            
    def __bias_force(self, simulation_temperature, potential_energy, md_force):
        Epot = potential_energy - self.its_parameters['its_energy_shift']
        simulation_beta = 1/(simulation_temperature*scipy.constants.value('Boltzmann constant in eV/K'))
        if os.path.isfile('MoREST_ITS_nk.npy'):
            n_k = np.loadtxt('MoREST_ITS_nk.npy')
        else:
            n_k = self.its_parameters['its_initial_nk']
        #print(n_k,type(n_k))
        bias_numerator = 0
        bias_denominator = 0
        for i,i_beta in enumerate(self.its_parameters['its_replica_beta']):
            #print(np.exp(-1*i_beta*Epot))
            bias_numerator += n_k[i]*i_beta*np.exp(-1*i_beta*Epot)
            bias_denominator += n_k[i]*np.exp(-1*i_beta*Epot)
        return md_force*bias_numerator/(simulation_beta*bias_denominator)
    
    def __pk_nk(self):
        potential_energy_list = np.loadtxt('MoREST_ITS_potential_energy.list') - self.its_parameters['its_energy_shift']
        if os.path.isfile('MoREST_ITS_nk.npy'):
            n_k = np.loadtxt('MoREST_ITS_nk.npy')
        else:
            n_k = self.its_parameters['its_initial_nk']
        P_k = []
        for i,i_beta in enumerate(self.its_parameters['its_replica_beta']):
            tmp_Pk = np.sum(np.exp(-1*i_beta*potential_energy_list))
            #print(tmp_Pk)
            P_k.append(tmp_Pk*n_k[i])
        P_k = np.array(P_k)
        p_k = P_k/np.sum(P_k)
        np.savetxt('MoREST_ITS_pk.npy',p_k)
        #print(p_k)
        return p_k,n_k
