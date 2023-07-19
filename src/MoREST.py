import os
import numpy as np
from glob import glob
from initialization import initialize_modules
from read_parameters import read_parameters
from trajectory_scattering import scattering_velocity_Verlet, scattering_Runge_Kutta_4th
from collective_variable import collective_variables


class morest:
    '''
    The Molecular Reaction Simulation Toolkits module. 
    '''

    def __init__(self, parameter_file='MoREST.in', calculator=None):
        '''
        Calculator is required, when many body potential is specified as 'on_the_fly'.
        '''
        self.calculator = calculator
        MoREST_parameters = read_parameters(parameter_file=parameter_file)
        self.morest_parameters = MoREST_parameters.get_morest_parameters()

        if self.morest_parameters['morest_initialization']:
            self.log_morest = open('MoREST.log','w', buffering=1)
            self.log_morest.write('-----------MoREST start to work-----------\n\n')
        else:
            self.log_morest = open('MoREST.log','a', buffering=1)
            self.log_morest.write('\n-----------MoREST continue to work--------\n\n')
    
        MoREST_parameters.write_morest_parameters(self.log_morest)
        self.log_morest.write('\n')

        super(morest, self).__init__(self.morest_parameters, self.calculator, MoREST_parameters)

        #################### Phase space sampling initialization ##############################
        if self.morest_parameters['phase_space_sampling']:
            initialize_modules.initialize_phase_space_sampling(MoREST_parameters)
    
        #################### Trajectory scattering initialization #############################
        if self.morest_parameters['trajectory_scattering']:
            initialize_modules.initialize_trajectory_scattering(MoREST_parameters)

        #################### Structure searching initialization ###############################
        if self.morest_parameters['structure_searching']:
            initialize_modules.initialize_structure_searching(MoREST_parameters)
    
        #################### Enhanced sampling initialization #################################
        if self.morest_parameters['enhanced_sampling']:
            initialize_modules.initialize_enhanced_sampling(MoREST_parameters)

        #################### Wall potential initialization ####################################
        if self.morest_parameters['wall_potential']:
            initialize_modules.initialize_wall_potential(MoREST_parameters)

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
                self.enhanced_sampling_re(simulation_maxsteps)
            elif self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['its'.upper()]:
                self.enhanced_sampling_its(simulation_maxsteps)
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
        simulation_maxsteps = self.scattering_parameters['scattering_traj_length']
        self.stop_condition = collective_variables(from_CVs_file=False, CVs_list=self.scattering_parameters['scattering_traj_stop'])
        i_traj = 0
        for i_traj in range(self.scattering_parameters['scattering_traj_number']):
            if self.scattering_parameters['scattering_method'].upper() in ['VV']:
                self.scattering_job = scattering_velocity_Verlet(self.morest_parameters, self.scattering_parameters, calculator=self.calculator, i_traj=i_traj, log_morest=self.log_morest)
            elif self.scattering_parameters['scattering_method'].upper() in ['RK4']:
                self.scattering_job = scattering_Runge_Kutta_4th(self.morest_parameters, self.scattering_parameters, calculator=self.calculator, i_traj=i_traj, log_morest=self.log_morest)
            else:
                self.log_morest.write('It is not clear which scattering method will be used.\n')
                self.log_morest.close()
                raise Exception('Which scattering method will you use?')
            current_step, current_system = self.scattering_job.current_step, self.scattering_job.current_system
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

    def structure_searching(self):
        '''
        This function is called to excute structure searching method.
        --------
        INPUT:
            calculator: The same as the calculator in ASE. It is required, when many body potential is specified as 'on_the_fly'.
        '''
        searching_convergence = self.searching_parameters['searching_convergence']
        searching_maxsteps = self.searching_parameters['searching_max_steps']
        if self.morest_parameters['enhanced_sampling']:
            self.morest_parameters['enhanced_sampling'] = False # TODO: enhanced sampling method for trajectory scattering
        else:
            current_convergence, current_step, current_system = self.searching_job.current_convergence, self.searching_job.current_step, self.searching_job.current_system
            if self.morest_parameters['wall_potential']:
                while current_convergence >= searching_convergence and current_step <= searching_maxsteps:
                    general_coordinate = current_system.get_positions()
                    bias_forces = self.wall_potential(general_coordinate)
                    current_convergence, current_step, current_system= self.searching_job.generate_new_step(bias_forces)
            else:
                while current_convergence >= searching_convergence and current_step <= searching_maxsteps:
                    current_convergence, current_step, current_system= self.searching_job.generate_new_step()
        self.log_morest.write('Structure optimization with '+self.searching_parameters['searching_method']+' method is finished!\n')
        self.mission_complete()

    def enhanced_sampling_re(self,simulation_maxsteps):
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
                    current_step[i], current_system[i] = i_sampling_job.generate_new_step(bias_forces,current_system[i])
                    current_potential_energy[i] = i_sampling_job.current_potential_energy
                else:
                    current_step[i], current_system[i] = i_sampling_job.generate_new_step(updated_current_system=current_system[i])
                    current_potential_energy[i] = i_sampling_job.current_potential_energy
        # --------------- (REMD) run ----------------------------------------------------------------
        while current_step[-1] <= simulation_maxsteps:
            for i,i_sampling_job in enumerate(self.sampling_job):
                if self.morest_parameters['wall_potential']:
                    general_coordinate = current_system[i].get_positions()
                    bias_forces = self.wall_potential(general_coordinate)
                    current_step[i], current_system[i] = i_sampling_job.generate_new_step(bias_forces,current_system[i])
                    current_potential_energy[i] = i_sampling_job.current_potential_energy
                else:
                    current_step[i], current_system[i] = i_sampling_job.generate_new_step(updated_current_system=current_system[i])
                    current_potential_energy[i] = i_sampling_job.current_potential_energy
            current_step, current_system = self.re_sampling.remd(current_step, current_potential_energy, current_system)

    def enhanced_sampling_its(self, simulation_maxsteps):
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
        current_step, current_system = self.sampling_job.current_step, self.sampling_job.current_system
        simulation_temperature = self.md_parameters['md_temperature']
        while current_step <= simulation_maxsteps:
            potential_energy = self.sampling_job.current_potential_energy
            md_forces = self.sampling_job.current_forces
            if self.its_sampling.its_if_converge():
                bias_forces_its = self.its_sampling.its_sampling(simulation_temperature, potential_energy, md_forces)
            else:
                bias_forces_its = self.its_sampling.its_optimization(simulation_temperature, potential_energy, \
                                        current_step, md_forces, self.log_morest)
            if self.morest_parameters['wall_potential']:
                general_coordinate = current_system.get_positions()
                bias_forces_wall_potential = self.wall_potential(general_coordinate)
                bias_forces = bias_forces_its + bias_forces_wall_potential
            else:
                bias_forces = bias_forces_its
            current_step, current_system = self.sampling_job.generate_new_step(bias_forces)


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
