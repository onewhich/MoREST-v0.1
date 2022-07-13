import os
import numpy as np
import scipy.constants

class its:
    '''
    The integrated tempering sampling module.
    '''
    
    def __init__(self, its_parameters):
        #self.its_parameters = np.load('MoREST_ITS_parameters.npy',allow_pickle=True).item()
        #self.log_its = open('MoREST_ITS.log','w')
        self.its_parameters = its_parameters
        
    def its_optimization(self, simulation_temperature, potential_energy, current_step, md_force, log_morest):
        #print(current_step)
        if current_step % self.its_parameters['its_trial_MD_steps'] == 0 :
            if current_step != 0 :
                #print('opting')
                #current_step = 0
                p_k, n_k = self.__pk_nk()
                
                log_morest.write('ITS optimization in '+str(current_step)+' steps.\n\n')
                log_morest.write('Current p_k:    ')
                for i_p in p_k:
                    log_morest.write(str(i_p)+'    ')
                log_morest.write('\n\n')
                log_morest.write('Current n_k:    ')
                for i_n in n_k:
                    log_morest.write(str(i_n)+'    ')
                log_morest.write('\n\n')
            
                new_nk = n_k * self.its_parameters['its_pk0'] / p_k
                #new_nk = n_k * np.sqrt(self.its_parameters['its_pk0'] / p_k)  # test
                new_nk /= np.sum(new_nk)
                np.savetxt('MoREST_ITS_nk.npy',new_nk)
                os.remove('MoREST_ITS_potential_energy.npy')
                return md_force - md_force # No bias forces return
                #bias_force = self.__bias_force(simulation_temperature, potential_energy, md_force)
                #return bias_force#, current_step
            else:
                return md_force - md_force # No bias forces return
        else:
            #print('not opting')
            with open('MoREST_ITS_potential_energy.npy','a') as potential_energy_list:
                potential_energy_list.write(str(potential_energy)+'\n')
            return md_force - md_force # No bias forces return
            '''
            try:
                potential_energy_list = []
                potential_energy_list.append(np.loadtxt('MoREST_ITS_potential_energy.npy'))
                print('list exist: read ',potential_energy_list)
                potential_energy_list = np.array(potential_energy_list)
                potential_energy_list = np.append(potential_energy_list, potential_energy)
                print('list exist: ',potential_energy_list)
                np.savetxt('MoREST_ITS_potential_energy.npy',potential_energy_list)
            except:
                potential_energy_list = []
                potential_energy_list.append(potential_energy)
                potential_energy_list = np.array(potential_energy_list)
                print('list not exist:',potential_energy_list)
                np.savetxt('MoREST_ITS_potential_energy.npy', potential_energy_list)
            '''
            '''
            potential_energy_list = []
            potential_energy_list.append(np.loadtxt('MoREST_ITS_potential_energy.npy'))
            potential_energy_list = np.array(potential_energy_list)
            potential_energy_list = np.append(potential_energy_list, potential_energy)
            print('list exist: ',potential_energy_list)
            np.savetxt('MoREST_ITS_potential_energy.npy',potential_energy_list)
            '''
            #bias_force = self.__bias_force(simulation_temperature, potential_energy, md_force)
            #return bias_force#, current_step
            
       
    def its_if_converge(self):
        #if not os.path.isfile('MoREST_ITS_pk.npy'):
        #    return False
        try:
            p_k = np.loadtxt('MoREST_ITS_pk.npy')
            #print('Debug: enhanced_sampling_ITS/its_if_converge: MoREST_ITS_pk is opened.')
        except:
            #print('Debug: enhanced_sampling_ITS/its_if_converge: MoREST_ITS_pk does not exist.')
            return False
        if abs(np.max(p_k - self.its_parameters['its_pk0'])) < self.its_parameters['its_delta_pk']:
            #if os.path.isfile('MoREST_ITS_potential_energy.npy'):
            try:
                os.remove('MoREST_ITS_potential_energy.npy')
            except:
                pass
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
        #if os.path.isfile('MoREST_ITS_nk.npy'):
        try:
            n_k = np.loadtxt('MoREST_ITS_nk.npy')
            #print('Debug: enhanced_sampling_ITS/__bias_force: MoREST_ITS_nk is opened.')
        #else:
        except:
            n_k = self.its_parameters['its_initial_nk']
            #print('Debug: enhanced_sampling_ITS/__bias_force: MoREST_ITS_nk does not exist.')
        #print(n_k,type(n_k))
        bias_numerator = 0
        bias_denominator = 0
        for i,i_beta in enumerate(self.its_parameters['its_replica_beta']):
            #print(np.exp(-1*i_beta*Epot))
            bias_numerator += n_k[i]*i_beta*np.exp(-1*i_beta*Epot)
            bias_denominator += n_k[i]*np.exp(-1*i_beta*Epot)
        return md_force*(bias_numerator/(simulation_beta*bias_denominator)-1) # substract original forces and return the pure bias forces
    
    def __pk_nk(self):
        if abs(self.its_parameters['its_energy_shift'] - 0.) > 1e-3:
            potential_energy_list = np.loadtxt('MoREST_ITS_potential_energy.npy') - self.its_parameters['its_energy_shift']
        else:
            tmp_potential_energy_list = np.loadtxt('MoREST_ITS_potential_energy.npy')
            potential_energy_list = tmp_potential_energy_list - np.min(tmp_potential_energy_list)
        #if os.path.isfile('MoREST_ITS_nk.npy'):
        try:
            n_k = np.loadtxt('MoREST_ITS_nk.npy')
            #print('Debug: enhanced_sampling_ITS/__pk_nk: MoREST_ITS_nk is opened.')
        #else:
        except:
            n_k = self.its_parameters['its_initial_nk']
            #print('Debug: enhanced_sampling_ITS/__pk_nk: MoREST_ITS_nk does not exist.')
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
