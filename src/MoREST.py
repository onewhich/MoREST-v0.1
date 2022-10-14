import os
import numpy as np
from glob import glob
from read_parameters import read_parameters
from phase_space_sampling import velocity_Verlet
from trajectory_scattering import scattering_velocity_Verlet, scattering_Runge_Kutta_4th, scattering_Runge_Kutta_4th_a
from enhanced_sampling import its, re
from wall_potential import repulsive_wall
from collective_variable import collective_variables


class morest:
    '''
    The Molecular Reaction Simulation Toolkits module. 
    '''

    def __init__(self, parameter_file='MoREST.in', calculator=None):
        '''
        Calculator is required, when many body potential is specified as 'on_the_fly'.
        '''
        MoREST_parameters = read_parameters(parameter_file=parameter_file)
        self.morest_parameters = MoREST_parameters.get_morest_parameters()

        if self.morest_parameters['morest_initialization']:
            self.log_morest = open('MoREST.log','w', buffering=1)
            self.log_morest.write('-----------MoREST start to work-----------\n\n')
        else:
            self.log_morest = open('MoREST.log','a', buffering=1)
            self.log_morest.write('\n-----------MoREST continue to work--------\n\n')
    
        #MoREST_parameters.write_parameters(self.log_morest)
        self.log_morest.write('\n')

        #################### Phase space sampling initialization ##############################
        if self.morest_parameters['phase_space_sampling']:
            if not self.morest_parameters['morest_load_parameters_file']:
                self.sampling_parameters = MoREST_parameters.get_sampling_parameters(self.log_morest)
                self.md_parameters = MoREST_parameters.get_md_parameters(self.log_morest)
            else:
                try:
                    self.sampling_parameters = np.load('MoREST_sampling_parameters.npy',allow_pickle=True).item()
                    self.md_parameters = np.load('MoREST_MD_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_sampling_parameters.npy, MoREST_MD_parameters.npy\n Read parameters from input file.\n\n')
                    self.sampling_parameters = MoREST_parameters.get_sampling_parameters(self.log_morest)
                    self.md_parameters = MoREST_parameters.get_md_parameters(self.log_morest)

            if self.sampling_parameters['sampling_initialization']:
                self.log_morest.write('Start to sample the phase space\n\n')
                #Method: '+str(self.sampling_parameters['sampling_method'])+'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
                try:
                    #os.remove('MoREST.str_new')
                    os.remove('MoREST_traj.xyz')
                    os.remove('MoREST_MD.log')
                except:
                    pass
            else:
                self.log_morest.write('Continue to sample the phase space\n\n')
                #Method: '+str(self.sampling_parameters['sampling_method'])+'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
            if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                if self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                    self.sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=calculator)
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_VR']:
                    self.sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=calculator, v_rescaling=True)
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Berendsen'.upper()]:
                    self.sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=calculator, Berendsen_rescaling=True)
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Langevin'.upper()]:
                    self.sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=calculator, Langevin_rescaling=True)
                elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_SVR']:
                    self.sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=calculator, sv_rescaling=True)
                else:
                    self.log_morest.write('It is not clear which ensemble will be used.\n')
                    self.log_morest.close()
                    raise Exception('Which ensemble will you use?')
            else:
                self.log_morest.write('It is not clear which sampling method will be used.\n')
                self.log_morest.close()
                raise Exception('Will you use the phase sampling method?')
    
        #################### Trajectory scattering initialization #############################
        if self.morest_parameters['trajectory_scattering']:
            if not self.morest_parameters['morest_load_parameters_file']:
                self.scattering_parameters = MoREST_parameters.get_scattering_parameters(self.log_morest)
            else:
                try:
                    self.scattering_parameters = np.load('MoREST_scattering_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_scattering_parameters.npy\n Read parameters from input file.\n\n')
                    self.scattering_parameters = MoREST_parameters.get_scattering_parameters(self.log_morest)

            if self.scattering_parameters['scattering_initialization']:
                self.log_morest.write('Start to sample the trajectories\n\n')
                try:
                    #os.remove('MoREST.str')
                    os.remove('MoREST_traj.xyz')
                    os.remove('MoREST_MD.log')
                except:
                    pass
            else:
                self.log_morest.write('Continue to sample the trajectories\n\n')
                
            self.stop_condition = collective_variables(from_CVs_file=False, CVs_list=self.scattering_parameters['scattering_traj_stop'])
            if self.scattering_parameters['scattering_method'].upper() in ['VV']:
                self.scattering_job = scattering_velocity_Verlet(self.morest_parameters, self.scattering_parameters, calculator=calculator)
            elif self.scattering_parameters['scattering_method'].upper() in ['RK4']:
                self.scattering_job = scattering_Runge_Kutta_4th(self.morest_parameters, self.scattering_parameters, calculator=calculator)
            else:
                    self.log_morest.write('It is not clear which scattering method will be used.\n')
                    self.log_morest.close()
                    raise Exception('Which scattering method will you use?')
    
        #################### Enhanced sampling initialization #################################
        if self.morest_parameters['enhanced_sampling']:
            if not self.morest_parameters['morest_load_parameters_file']:
                self.enhanced_sampling_parameters = MoREST_parameters.get_enhanced_sampling_parameters(self.log_morest)
            else:
                try:
                    self.enhanced_sampling_parameters = np.load('MoREST_enhanced_sampling_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_enhanced_sampling_parameters.npy\n Read parameters from input file.\n\n')
                    self.enhanced_sampling_parameters = MoREST_parameters.get_enhanced_sampling_parameters(self.log_morest)
            #for key in self.enhanced_sampling_parameters:
            #    print(key+' : '+str(self.enhanced_sampling_parameters[key]))
        
            self.log_morest.write('Enahanced sampling method \"'+str(self.enhanced_sampling_parameters['enhanced_sampling_method'])+'\" is called.\n\n')
            if self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['re'.upper()]:
                if not self.morest_parameters['morest_load_parameters_file']:
                    self.re_parameters = MoREST_parameters.get_re_parameters(self.log_morest)
                else:
                    try:
                        self.re_parameters = np.load('MoREST_RE_parameters.npy',allow_pickle=True).item()
                    except:
                        self.log_morest.write('Can not find parameters files: MoREST_RE_parameters.npy\n Read parameters from input file.\n\n')
                        self.re_parameters = MoREST_parameters.get_re_parameters(self.log_morest)

                re_file_name_title = 'MoREST_RE_'
                if self.re_parameters['re_initialization']:
                    try:
                        for re_file in glob('./'+re_file_name_title+'*'):
                            os.remove(re_file)
                        #os.remove(re_file_name_title+'replica_index.log')
                        #for i,T in enumerate(self.re_parameters['re_replica_temperatures']):
                        #    os.remove(re_file_name_title+str(T)+'K.log')
                        #    os.remove(re_file_name_title+str(T)+'K_traj.xyz')
                        #    #os.remove(re_file_name_title+'replica_'+str(i)+'.log')
                        #    #os.remove(re_file_name_title+'replica_'+str(i)+'_traj.xyz')
                    except:
                        pass
                    self.log_morest.write('Replica exchange method is initialized.\n\n')
                self.re_sampling = re(self.re_parameters)
                molecules = self.re_sampling.get_current_molecules()
                if len(molecules) != len(self.re_parameters['re_replica_temperatures']):
                    self.log_morest.write('The number of structures do not fit the number of temperatures.\n\n')
                    raise Exception('The number of structures do not fit the number of temperatures.')
                log_file_name = self.re_sampling.get_log_file_name()
                traj_file_name = self.re_sampling.get_traj_file_name()
                self.sampling_job = []
                for i,T in enumerate(self.re_parameters['re_replica_temperatures']):
                    if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                        if self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                            tmp_sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, \
                                                                molecules[i], log_file_name[i], traj_file_name[i], T, calculator=calculator)
                        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_VR']:
                            tmp_sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, \
                                                                molecules[i], log_file_name[i], traj_file_name[i], T, calculator=calculator, v_rescaling=True)
                        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Berendsen'.upper()]:
                            tmp_sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, \
                                                                molecules[i], log_file_name[i], traj_file_name[i], T, calculator=calculator, Berendsen_rescaling=True)
                        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_Langevin'.upper()]:
                            tmp_sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, \
                                                                molecules[i], log_file_name[i], traj_file_name[i], T, calculator=calculator, Langevin_rescaling=True)
                        elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVT_SVR']:
                            tmp_sampling_job = velocity_Verlet(self.morest_parameters, self.sampling_parameters, self.md_parameters, \
                                                                molecules[i], log_file_name[i], traj_file_name[i], T, calculator=calculator, sv_rescaling=True)
                    self.sampling_job.append(tmp_sampling_job)
                    self.log_morest.write('Replica '+str(i)+' at '+str(T)+' K is ready.\n')
                self.log_morest.write('\n')
                    
            elif self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['its'.upper()]:
                if not self.morest_parameters['morest_load_parameters_file']:
                    self.its_parameters = MoREST_parameters.get_its_parameters(self.log_morest)
                else:
                    try:
                        self.its_parameters = np.load('MoREST_ITS_parameters.npy',allow_pickle=True).item()
                    except:
                        self.log_morest.write('Can not find parameters files: MoREST_ITS_parameters.npy\n Read parameters from input file.\n\n')
                        self.its_parameters = MoREST_parameters.get_its_parameters(self.log_morest)
                if self.its_parameters['its_initialization']:
                    try:
                        os.remove('MoREST_ITS_pk.npy')
                        os.remove('MoREST_ITS_nk.npy')
                        os.remove('MoREST_ITS_potential_energy.npy')
                    except:
                        pass
                    self.log_morest.write('Integrated tempering sampling method is initialized.\n\n')
                self.its_sampling = its(self.its_parameters)
            else:
                self.log_morest.write('It is not clear which enhanced sampling method will be used.\n')
                self.log_morest.close()
                raise Exception('Which enhanced sampling method will you use?')
        #    for key in self.its_parameters:
        #        print(key+' : '+str(self.its_parameters[key]))

        #################### Wall potential initialization ####################################
        if self.morest_parameters['wall_potential']:
            if not self.morest_parameters['morest_load_parameters_file']:
                self.wall_potential_parameters = MoREST_parameters.get_wall_potential_parameters(self.log_morest)
            else:
                try:
                    self.wall_potential_parameters = np.load('MoREST_wall_potential_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_wall_potential_parameters.npy\n Read parameters from input file.\n\n')
                    self.wall_potential_parameters = MoREST_parameters.get_wall_potential_parameters(self.log_morest)
            #for key in self.wall_potential_parameters:
            #    print(key+' : '+str(self.wall_potential_parameters[key]))
            self.log_morest.write('Wall potential \"'+str(self.wall_potential_parameters['wall_shape'])+' '\
                +str(self.wall_potential_parameters['wall_type'])+'\" is called.\n\n')
            self.wall = repulsive_wall(self.wall_potential_parameters)

        #MoREST_parameters.write_parameters(self.log_morest)

        
    def phase_space_sampling(self):
        '''
        This function is called to excute phase space sampling method.
        --------
        INPUT:
            calculator: The same as the calculator in ASE. It is required, when many body potential is specified as 'on_the_fly'.
        '''
        simulation_maxsteps = int(self.md_parameters['md_simulation_time']/self.md_parameters['md_time_step']) + 1
        if self.morest_parameters['enhanced_sampling']:
            if self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['re'.upper()]:
                current_step = []
                current_system = []
                current_potential_energy = []
                for i_sampling_job in self.sampling_job:
                    current_step.append(i_sampling_job.current_step)
                    current_system.append(i_sampling_job.current_system)
                    current_potential_energy.append(i_sampling_job.current_potential_energy)
                current_step = np.array(current_step)
                current_potential_energy = np.array(current_potential_energy)
                current_max_step = np.max(current_step)
                # --------------- (REMD) syncrhronize all replica to the same MD steps ----------------------
                for i,i_sampling_job in enumerate(self.sampling_job):
                    while current_step[i] < current_max_step:
                        if self.morest_parameters['wall_potential']:
                            general_coordinate = current_system[i].get_positions()
                            bias_forces = self.wall_potential(general_coordinate)
                            current_step[i], current_system[i] = i_sampling_job.generate_new_step(bias_forces)
                            current_potential_energy[i] = i_sampling_job.current_potential_energy
                        else:
                            current_step[i], current_system[i] = i_sampling_job.generate_new_step()
                            current_potential_energy[i] = i_sampling_job.current_potential_energy
                # --------------- (REMD) run ----------------------------------------------------------------
                while current_step[-1] <= simulation_maxsteps:
                    for i,i_sampling_job in enumerate(self.sampling_job):
                        if self.morest_parameters['wall_potential']:
                            general_coordinate = current_system[i].get_positions()
                            bias_forces = self.wall_potential(general_coordinate)
                            current_step[i], current_system[i] = i_sampling_job.generate_new_step(bias_forces)
                            current_potential_energy[i] = i_sampling_job.current_potential_energy
                        else:
                            current_step[i], current_system[i] = i_sampling_job.generate_new_step()
                            current_potential_energy[i] = i_sampling_job.current_potential_energy
                    current_step, current_system = self.re_sampling.remd(current_step, current_potential_energy, current_system)

            elif self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['its'.upper()]:
                current_step, current_system = self.sampling_job.current_step, self.sampling_job.current_system
                while current_step <= simulation_maxsteps:
                    bias_its_forces = self.enhanced_sampling_its(current_step)
                    if self.morest_parameters['wall_potential']:
                        general_coordinate = current_system.get_positions()
                        bias_forces_wall_potential = self.wall_potential(general_coordinate)
                        bias_forces = bias_its_forces + bias_forces_wall_potential
                    else:
                        bias_forces = bias_its_forces
                    current_step, current_system = self.sampling_job.generate_new_step(bias_forces)

        else:
            current_step, current_system = self.sampling_job.current_step, self.sampling_job.current_system
            if self.morest_parameters['wall_potential']:
                while current_step <= simulation_maxsteps:
                    general_coordinate = current_system.get_positions()
                    bias_forces = self.wall_potential(general_coordinate)
                    current_step, current_system= self.sampling_job.generate_new_step(bias_forces)
            else:
                while current_step <= simulation_maxsteps:
                    current_step, current_system= self.sampling_job.generate_new_step()
        self.log_morest.write('Phase space sampling with molecular dynamics method is finished!\n')
        self.mission_complete()

    def trajectory_scattering(self):
        '''
        This function is called to excute trajectory scattering method.
        --------
        INPUT:
            calculator: The same as the calculator in ASE. It is required, when many body potential is specified as 'on_the_fly'.
        '''
        current_step, current_system = self.scattering_job.current_step, self.scattering_job.current_system
        simulation_maxsteps = self.scattering_parameters['scattering_traj_length']
        while current_step <= simulation_maxsteps:
            if self.morest_parameters['enhanced_sampling']:
                self.morest_parameters['enhanced_sampling'] = False # TODO: enhanced sampling method for trajectory scattering
            else:
                if self.morest_parameters['wall_potential']:
                    general_coordinate = current_system.get_positions()
                    bias_forces = self.wall_potential(general_coordinate)
                    current_step, current_system= self.scattering_job.generate_new_step(bias_forces)
                else:
                    current_step, current_system= self.scattering_job.generate_new_step()
            if self.stop_condition.check_CVs_one(current_system):
                break
        self.log_morest.write('Trajectory scattering with molecular dynamics method is finished!\n')
        self.mission_complete()

    def enhanced_sampling_its(self, current_step):
        '''
        This function is called to excute enhanced sampling by phase space sampling module.
        --------
        INPUT:
            enhanced_sampling_method: Specify the method will be used, e.g., ITS, REMD...
            #if_initial:               Specify whether this API is called for the first time.
            simulation_temperature:   The temperature used in MD/MC module. Unit: K.
            simulation_maxsteps:      The simulation length, which is specified in number of steps, in MD/MC module.
            time_step:                The time step in MD/MC module. Unit: ps.
            potential_energy:         The potential energy of current configuration from MD/MC module.
            current_step:          This number is used to check whether trial MD is finished.
            md_forces:                 The forces vector of current configuration from MD/MC module.
        OUTPUT:
            bias_forces:           The bias forces vector generated by ITS and returned to MD/MC module. 
            #current_step:          The current MD step is returned to identify the number of opt and sampling steps in ITS
        '''
        simulation_temperature = self.md_parameters['md_temperature']
        potential_energy = self.sampling_job.current_potential_energy
        md_forces = self.sampling_job.current_forces
        if self.its_sampling.its_if_converge():
            bias_forces = self.its_sampling.its_sampling(simulation_temperature, potential_energy, md_forces) 
            return bias_forces
        else:
            bias_forces = self.its_sampling.its_optimization(simulation_temperature, potential_energy, \
                                    current_step, md_forces, self.log_morest)
            return bias_forces

    def wall_potential(self, general_coordinate):
        '''
        This function will read the positions of atoms and then add forces of the potential on the atoms.
        --------
        INPUT:
            general_coordinate:       The positions vector of atoms in the system.
        OUTPUT:
            wall_forces:            The forces of the wall potential on the atoms
        '''
        #self.log_morest.write('Debug: In wall potential \n')
        wall_forces = []
        wall_potential = []
        for i_coordinate in general_coordinate:
            i_wall_force, i_wall_potential = self.wall.get_repulsive_wall_force_potential(i_coordinate)
            wall_forces.append(i_wall_force)
            wall_potential.append(i_wall_potential)
        return np.array(wall_forces)


    def mission_complete(self):
        self.log_morest.write('\nThe mission of MoREST is complete!\n')
        self.log_morest.close()
