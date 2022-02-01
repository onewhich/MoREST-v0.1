import os, sys
import numpy as np
from read_parameters import read_parameters
from phase_space_sampling import velocity_Verlet
from enhanced_sampling import its
from wall_potential import opaque_wall, translucent_wall


class morest:
    '''
    The Molecular Reaction Simulation Toolkits module.
    '''

    def __init__(self, __parameter_file='MoREST.in'):
        self.__log_morest = open('MoREST.log','a')
        MoREST_parameters = read_parameters(log_morest=self.__log_morest, parameter_file=__parameter_file)
        self.morest_parameters = MoREST_parameters.get_morest_parameters()
        if self.morest_parameters['morest_api_fortran']:
            self.morest_parameters['morest_initialization'] = False

        if self.morest_parameters['morest_initialization']:
                self.__log_morest.close()
                self.__log_morest = open('MoREST.log','w')
                self.__log_morest.write('-----------MoREST start to work-----------\n\n')
        else:
            if self.morest_parameters['morest_api_fortran']:
                try:
                    if_restart = np.loadtxt('restart')
                except:
                    self.__log_morest.close()
                    self.__log_morest = open('MoREST.log','w')
                    self.__log_morest.write('-----------MoREST start to work-----------\n\n')
                    np.savetxt('restart', [1])
            else:
                self.__log_morest.write('\n-----------MoREST continue to work--------\n\n')

        #################### Phase space sampling initialization ##############################
        self.sampling_parameters = MoREST_parameters.get_sampling_parameters()
        self.md_parameters = MoREST_parameters.get_md_parameters()
        if self.sampling_parameters['phase_space_sampling']:
            if self.sampling_parameters['sampling_initialization']:
                self.__log_morest.write('Start to sample the phase space\nMethod: '+str(self.sampling_parameters['sampling_method'])\
                                       +'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
                try:
                    os.remove('MoREST.str_new')
                    os.remove('MoREST_traj.xyz')
                except:
                    pass
            else:
                self.__log_morest.write('Continue to sample the phase space\nMethod: '+str(self.sampling_parameters['sampling_method'])\
                                       +'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
                
        
        #################### Enhanced sampling initialization #################################
        self.enhanced_sampling_parameters = MoREST_parameters.get_enhanced_sampling_parameters()
        #for key in self.enhanced_sampling_parameters:
        #    print(key+' : '+str(self.enhanced_sampling_parameters[key]))
        if self.enhanced_sampling_parameters['enhanced_sampling']:
            if not self.morest_parameters['morest_api_fortran']:
                self.__log_morest.write('Enahanced sampling method \"'+\
                    str(self.enhanced_sampling_parameters['enhanced_sampling_method'])+'\" is called:\n')
            if self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['its'.upper()]:
                if self.morest_parameters['morest_initialization']:
                    self.its_parameters = MoREST_parameters.get_its_parameters(self.__log_morest)
                else:
                    try:
                        self.its_parameters = np.load('MoREST_ITS_parameters.npy',allow_pickle=True).item()
                    except:
                        self.its_parameters = MoREST_parameters.get_its_parameters(self.__log_morest)
        #    for key in self.its_parameters:
        #        print(key+' : '+str(self.its_parameters[key]))

        #################### Wall potential initialization ####################################
        self.wall_potential_parameters = MoREST_parameters.get_wall_potential_parameters()
        #for key in self.wall_potential_parameters:
        #    print(key+' : '+str(self.wall_potential_parameters[key]))
        if self.wall_potential_parameters['wall_potential']:
            if not self.morest_parameters['morest_api_fortran']:
                self.__log_morest.write('Wall potential \"'+\
                    str(self.wall_potential_parameters['wall_type'])+'\" is called:\n')
            if self.wall_potential_parameters['wall_type'].upper() in ['Plane_opaque_wall'.upper(),\
                                                    'Plane_translucent_wall'.upper()]:
                if self.morest_parameters['morest_initialization']:
                    self.plane_wall_parameters = MoREST_parameters.get_plane_wall_parameters()
                else:
                    try:
                        self.plane_wall_parameters = np.load('MoREST_plane_wall_parameters.npy',allow_pickle=True).item()
                    except:
                        self.plane_wall_parameters = MoREST_parameters.get_plane_wall_parameters()
            if self.wall_potential_parameters['wall_type'].upper() in ['Spherical_opaque_wall'.upper(),\
                                                    'Spherical_translucent_wall'.upper()]:
                if self.morest_parameters['morest_initialization']:
                    self.spherical_wall_parameters = MoREST_parameters.get_spherical_wall_parameters()
                else:
                    try:
                        self.spherical_wall_parameters = np.load('MoREST_spherical_wall_parameters.npy',allow_pickle=True).item()
                    except:
                        self.spherical_wall_parameters = MoREST_parameters.get_spherical_wall_parameters()
            if self.wall_potential_parameters['wall_type'].upper() in ['Plane_opaque_wall'.upper(),\
                                                    'Spherical_opaque_wall'.upper()]:
                self.wall = opaque_wall()
            elif self.wall_potential_parameters['wall_type'].upper() in ['Plane_translucent_wall'.upper(),\
                                                    'Spherical_translucent_wall'.upper()]:
                self.wall = translucent_wall()
        #    for key in self.plane_wall_parameters:
        #        print(key+' : '+str(self.plane_wall_parameters[key]))

        
    def phase_space_sampling(self, calculator=None):
        '''
        This function is called to excute phase space sampling method.
        --------
        '''
        if self.sampling_parameters['sampling_method'].upper() in ['MD'] and self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
            sampling_job = velocity_Verlet(self.sampling_parameters, self.md_parameters, calculator)
        else:
            __log_morest.write('It is not clear which sampling method and ensemble will be used.\n')
            __log_morest.close()
            raise Exception('Will you use the phase sampling method?')
        current_step, current_system = sampling_job.get_current_structure()
        max_time_step = int(self.md_parameters['md_simulation_time']/self.md_parameters['md_time_step']) + 1
        for i_step in range(current_step, max_time_step):
            if self.wall_potential_parameters['wall_potential']:
                general_coordinate = current_system.get_positions()
                bias_force_wall_potential = self.wall_potential(general_coordinate)
            else:
                bias_force_wall_potential = None
            current_system= sampling_job.generate_new_step(bias_force_wall_potential)
        self.__log_morest.write('Phase space sampling with molecular dynamics method in microcanonical ensemble is finished!\n')
        self.mission_complete()
    
    def bias_sampling(self, simulation_temperature, simulation_maxsteps, \
                   time_step, potential_energy, current_md_step, md_force, general_coordinate):
        '''
        This function combines enhanced_sampling and wall_potential together to be used by MD/MC module.
        --------
        INPUT:
            The same as the enhanced_sampling function and the wall_potential function
        OUTPUT:
            bias_force: The bias forces are combined from enhanced sampling and wall potential.
        '''
        
        if_call_enhanced_sampling = False
        if_call_wall_potential = False

        #self.__log_morest.write('Debug: calling bias sampling\n')
        #self.__log_morest.write('Debug: MD step: '+str(current_md_step)+'\n')

        if self.enhanced_sampling_parameters['enhanced_sampling']:
            #self.__log_morest.write('Debug: calling enhanced sampling\n')
            bias_force_enhanced_sampling = self.enhanced_sampling(simulation_temperature, simulation_maxsteps, \
                                 time_step, potential_energy, current_md_step, md_force)
            if_call_enhanced_sampling = True
            #print(bias_force_enhanced_sampling)
        if self.wall_potential_parameters['wall_potential']:
            #self.__log_morest.write('Debug: calling wall potential\n')
            bias_force_wall_potential = self.wall_potential(general_coordinate)
            if_call_wall_potential = True
            #print(bias_force_wall_potential)
            
        if if_call_enhanced_sampling and if_call_wall_potential:
            return bias_force_enhanced_sampling + bias_force_wall_potential
        elif if_call_enhanced_sampling:
            return bias_force_enhanced_sampling
        elif if_call_wall_potential:
            return bias_force_wall_potential
        else:
            self.__log_morest.write('Both enhanced sampling and wall potential do not work.\n')
            self.__log_morest.close()
            return np.array([0])
    
    
    def enhanced_sampling(self, simulation_temperature, simulation_maxsteps, \
                   time_step, potential_energy, current_md_step, md_force):
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
            current_md_step:          This number is used to check whether trial MD is finished.
            md_force:                 The force vector of current configuration from MD/MC module.
        OUTPUT:
            bias_force:           The bias force vector generated by ITS and returned to MD/MC module. 
            #current_md_step:          The current MD step is returned to identify the number of opt and sampling steps in ITS
        '''

        if self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['its'.upper()]:
            #self.__log_morest.write('Debug: In ITS sampling\n')
            #self.__log_morest.write('Debug: ITS MD step: '+str(current_md_step)+'\n')
            '''
            if if_initial or ( if_initial == 1 ):
                #if os.path.isfile('MoREST_ITS_pk.npy'):
                os.remove('MoREST_ITS_pk.npy')
                #if os.path.isfile('MoREST_ITS_nk.npy'):
                os.remove('MoREST_ITS_nk.npy')
                #if os.path.isfile('MoREST_ITS_potential_energy.npy'):
                os.remove('MoREST_ITS_potential_energy.npy')

                self.__log_morest.write('Start to initialize integrated tempering sampling method.\n')
                #for key in self.its_parameters:
                #    self.__log_morest.write(key+' : '+str(self.its_parameters[key])+'\n')
                #self.__log_morest.write('\n')

                bias_force = its().its_optimization(simulation_temperature,\
                                        potential_energy, current_md_step,\
                                        md_force, self.__log_morest)
                return bias_force

            elif ( not if_initial ) or ( if_initial == 0 ):
                if its().its_if_converge():
                    bias_force = its().its_sampling(simulation_temperature, potential_energy, md_force) 
                    return bias_force
                else:
                    bias_force = its().its_optimization(simulation_temperature, potential_energy, \
                                            current_md_step, md_force, self.__log_morest)
                    return bias_force
            '''
            if its().its_if_converge():
                bias_force = its().its_sampling(simulation_temperature, potential_energy, md_force) 
                return bias_force
            else:
                bias_force = its().its_optimization(simulation_temperature, potential_energy, \
                                        current_md_step, md_force, self.__log_morest)
                return bias_force
        else:
            self.__log_morest.write('No enhanced sampling method was matched.\n')
            self.__log_morest.close()
            return np.array([0])

        
        
    def wall_potential(self, general_coordinate):
        '''
        This function will read the positions of atoms and then add forces of the potential on the atoms.
        --------
        INPUT:
            general_coordinate:       The positions vector of atoms in the system.
        OUTPUT:
            wall_force:            The forces of the wall potential on the atoms
        '''

        if not self.wall_potential_parameters['collective_variable']:
            #self.__log_morest.write('Debug: In wall potential \n')
            if self.wall_potential_parameters['wall_type'].upper() in ['Plane_opaque_wall'.upper()]:
                #self.__log_morest.write('\n')
                #self.__log_morest.write('The plane opaque wall potential and force on atoms: XYZ coordinate, Potential, Forces\n')
                wall_force = []
                for i_coordinate in general_coordinate:
                    i_wall_force, i_wall_potential = self.wall.get_plane_opaque_wall_force_potential(i_coordinate)
                    wall_force.append(i_wall_force)
                    #self.__log_morest.write(str(i_coordinate)+' , '+str(i_wall_potential)+' , '+str(i_wall_force))
                    #self.__log_morest.write('\n')
                #self.__log_morest.write('\n')
                return np.array(wall_force)
            
            if self.wall_potential_parameters['wall_type'].upper() in ['Spherical_opaque_wall'.upper()]:
                wall_force = []
                for i_coordinate in general_coordinate:
                    i_wall_force, i_wall_potential = self.wall.get_spherical_opaque_wall_force_potential(i_coordinate)
                    wall_force.append(i_wall_force)
                return np.array(wall_force)

            if self.wall_potential_parameters['wall_type'].upper() in ['Plane_translucent_wall'.upper()]:
                #self.__log_morest.write('\n')
                #self.__log_morest.write('The plane translucent wall potential and force on atoms: XYZ coordinate, Potential, Forces\n')
                wall_force = []
                for i_coordinate in general_coordinate:
                    i_wall_force, i_wall_potential = self.wall.get_plane_translucent_wall_force_potential(i_coordinate)
                    wall_force.append(i_wall_force)
                    #self.__log_morest.write(str(i_coordinate)+' , '+str(i_wall_potential)+' , '+str(i_wall_force))
                    #self.__log_morest.write('\n')
                #self.__log_morest.write('\n')
                return np.array(wall_force)
            
            if self.wall_potential_parameters['wall_type'].upper() in ['Spherical_translucent_wall'.upper()]:
                wall_force = []
                for i_coordinate in general_coordinate:
                    i_wall_force, i_wall_potential = self.wall.get_spherical_translucent_wall_force_potential(i_coordinate)
                    wall_force.append(i_wall_force)
                return np.array(wall_force)

            else:
                self.__log_morest.write('No wall type was matched.\n')
                self.__log_morest.close()
                return np.array([0])

        else:
            general_coordinate = CV_to_xyz(general_coordinate) # TODO conversion function is not exist.


    def mission_complete(self):
        self.__log_morest.write('\nThe mission of MoREST is complete!\n')
        self.__log_morest.close()