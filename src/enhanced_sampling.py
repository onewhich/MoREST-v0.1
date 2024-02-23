import os
import numpy as np
#import scipy.constants
from ase import units
from structure_io import read_xyz_file, read_xyz_traj

class re:
    '''
    The replica exchange molecular dynamics / Monte Carlo module
    Y. Sugita, Y. Okamotor Chemical Physics Letters 314 (1999) 141–151
    '''
    
    def __init__(self, re_parameters):
        self.re_parameters = re_parameters
        self.file_name_title = 'MoREST_RE_'

        self.replica_index_file = open(self.file_name_title+'replica_index.log','a',buffering=1)
        if self.re_parameters['re_initialization']:
            self.replica_index = np.arange(self.re_parameters['re_number_of_replica'])
            self.write_replica_index()
        else:
            try:
                self.replica_index = np.loadtxt(self.file_name_title+'replica_index.log',dtype='int')[-1]
            except:
                self.replica_index = np.arange(self.re_parameters['re_number_of_replica'])
                self.write_replica_index()
        
        self.temperature_log_file_list = []
        self.temperature_traj_file_name_list = []
        #self.replica_log_file_list = []
        #self.replica_traj_file_name_list = []
        for i,T in enumerate(self.re_parameters['re_replica_temperatures']):
            self.temperature_log_file_list.append(self.file_name_title+str(T)+'K.log')
            self.temperature_traj_file_name_list.append(self.file_name_title+str(T)+'K_traj.xyz')
            #self.replica_log_file_list.append(self.file_name_title+'replica_'+str(i)+'log')
            #self.replica_traj_file_name_list.append(self.file_name_title+'replica_'+str(i)+'_traj.xyz')

    def get_current_molecules(self):
        if self.re_parameters['re_initialization']:
            if self.re_parameters['re_multiple_initi_structures']:
                current_molecules = read_xyz_traj(self.re_parameters['re_init_structures_list'])
            else:
                current_molecules = []
                for i in range(self.re_parameters['re_number_of_replica']):
                    current_molecules.append(None)
        else:
            try:
                current_molecules = []
                for i in range(self.re_parameters['re_number_of_replica']):
                    current_molecules.append(read_xyz_file(self.temperature_traj_file_name_list[i]))
            except:
                if self.re_parameters['re_multiple_initi_structures']:
                    current_molecules = read_xyz_traj(self.re_parameters['re_init_structures_list'])
                else:
                    current_molecules = []
                    for i in range(self.re_parameters['re_number_of_replica']):
                        current_molecules.append(None)
        return current_molecules

    def get_log_file_name(self):
        return self.temperature_log_file_list

    def get_traj_file_name(self):
        return self.temperature_traj_file_name_list

    def write_replica_index(self):
        self.replica_index_file.write(str(self.replica_index[0]))
        for i in range(1, self.re_parameters['re_number_of_replica']):
            self.replica_index_file.write('    '+str(self.replica_index[i]))
        self.replica_index_file.write('\n')

    def remd_swap(self, i, current_step, current_system):
        T_replica = self.re_parameters['re_replica_temperatures']
        current_system[i].set_momenta(np.sqrt(T_replica[i+1]/T_replica[i]) * current_system[i].get_momenta())
        current_system[i+1].set_momenta(np.sqrt(T_replica[i]/T_replica[i+1]) * current_system[i+1].get_momenta())

        current_step[i], current_step[i+1] = current_step[i+1], current_step[i]
        current_system[i], current_system[i+1] = current_system[i+1], current_system[i]
        self.replica_index[i], self.replica_index[i+1] = self.replica_index[i+1], self.replica_index[i]
        return current_step, current_system

    def remd(self, current_step, current_potential_energy, current_system):
        if current_step[-1] % self.re_parameters['re_swap_interval'] == 0:
            current_step = np.array(current_step)
            current_potential_energy = np.array(current_potential_energy)
            replica_beta = self.re_parameters['re_replica_beta']
            self.re_parameters['re_current_swap_step'] = (current_step/self.re_parameters['re_swap_interval']).astype(int)
            starting_index = self.re_parameters['re_current_swap_step'][-1] % 2
            for i in range(starting_index, self.re_parameters['re_number_of_replica']-1, 2):
                delta = (replica_beta[i+1] - replica_beta[i]) * (current_potential_energy[i] - current_potential_energy[i+1])
                if delta <= 0:
                    #p_swap = 1
                    current_step, current_system = self.remd_swap(i, current_step, current_system)
                else:
                    p_swap = np.exp(-delta)
                    if p_swap >= np.random.random():
                        current_step, current_system = self.remd_swap(i, current_step, current_system)
            self.write_replica_index()
        return current_step, current_system

class its:
    '''
    The integrated tempering sampling module.
    Yi Qin Gao The Journal of Chemical Physics 128, 064105 (2008);
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
                            
                new_nk =  self.its_parameters['its_weight_pk'] * n_k * self.its_parameters['its_pk0'] / p_k  # can lead to p_k n_k SCF oscillation
                #new_nk = self.its_parameters['its_weight_pk'] * n_k * np.sqrt(self.its_parameters['its_pk0'] / p_k)  # test for p_k n_k SCF convergence
                new_nk /= np.sum(new_nk)
                np.savetxt('MoREST_ITS_nk.npy',new_nk)
                os.remove('MoREST_ITS_potential_energy.npy')

                log_morest.write('ITS optimization in '+str(current_step)+' steps.\n\n')
                log_morest.write('Current p_k:    ')
                for i_p in p_k:
                    log_morest.write(str(i_p)+'    ')
                log_morest.write('\n\n')
                log_morest.write('Current n_k:    ')
                for i_n in n_k:
                    log_morest.write(str(i_n)+'    ')
                log_morest.write('\n\n')

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
        if abs(np.max(p_k - self.its_parameters['its_pk0'])) < self.its_parameters['its_criteria_pk']:
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
        #simulation_beta = 1/(simulation_temperature*scipy.constants.value('Boltzmann constant in eV/K'))
        simulation_beta = 1/(simulation_temperature*units.kB)
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

class rp:
    '''
    The ring polymer molecular dynamics module.
    Annu. Rev. Phys. Chem. 2013. 64:387-413
    J. Chem. Phys. 133, 124104 (2010)
    '''
    def __init__(self, rp_parameters) -> None:
        pass

