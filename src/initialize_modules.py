import os
from glob import glob
import numpy as np
from molecular_dynamics_sampling import NVE_VV, NVK_VR, NVT_Berendsen, NVT_Langevin, NVT_SVR, NPH_SVR, NPT_Berendsen, NPT_Langevin, NPT_SVR
from path_integral_MD_sampling import RP_NVE_VV, RP_NVK_VR, RP_NVT_Berendsen, RP_NVT_Langevin, RP_NVT_SVR
from path_integral_MD_sampling_normal_mode import RP_NVE_VV_normal_mode, RP_NVK_VR_normal_mode, RP_NVT_Berendsen_normal_mode, RP_NVT_Langevin_normal_mode, RP_NVT_SVR_normal_mode
from molecular_dynamics_scattering import scattering_velocity_Verlet, scattering_Suzuki_Yoshida_4th, scattering_Runge_Kutta_4th, scattering_Langevin_dynamics
from molecule_rovibrating import rovibrating_velocity_Verlet, rovibrating_Suzuki_Yoshida_4th, rovibrating_Runge_Kutta_4th
from structure_searching import gradient_descent, L_BFGS_descent, scipy_L_BFGS_B_descent, FIRE_velocity_Verlet, BFGS_TS, L_BFGS_TS, dimer, GAD_velocity_Verlet
from enhanced_sampling import replica_exchange, integrated_tempering_sampling
from wall_potential import repulsive_wall
            
class initialize_modules:
    def __init__(self, morest_parameters, calculator, log_morest):
        self.morest_parameters = morest_parameters
        self.calculator = calculator
        self.log_morest = log_morest
        
    def initialize_phase_space_sampling(self, MoREST_parameters):
        if not self.morest_parameters['morest_load_parameters_file']:
            self.sampling_parameters = MoREST_parameters.get_sampling_parameters(self.log_morest)
            if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                self.MD_parameters = MoREST_parameters.get_MD_parameters(self.log_morest)
            elif self.sampling_parameters['sampling_method'].upper() in ['RPMD', 'RPMD_NM']:
                self.RPMD_parameters = MoREST_parameters.get_RPMD_parameters(self.log_morest)
        else:
            try:
                self.sampling_parameters = np.load('MoREST_sampling_parameters.npy',allow_pickle=True).item()
                if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                    self.MD_parameters = np.load('MoREST_MD_parameters.npy',allow_pickle=True).item()
                elif self.sampling_parameters['sampling_method'].upper() in ['RPMD', 'RPMD_NM']:
                    self.RPMD_parameters = np.load('MoREST_RPMD_parameters.npy',allow_pickle=True).item()
            except:
                self.log_morest.write('Can not find parameters files: MoREST_sampling_parameters.npy, MoREST_MD_parameters.npy, MoREST_RPMD_parameters.npy\n \
                                      Read parameters from input file.\n\n')
                self.sampling_parameters = MoREST_parameters.get_sampling_parameters(self.log_morest)
                if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                    self.MD_parameters = MoREST_parameters.get_MD_parameters(self.log_morest)
                elif self.sampling_parameters['sampling_method'].upper() in ['RPMD', 'RPMD_NM']:
                    self.RPMD_parameters = MoREST_parameters.get_RPMD_parameters(self.log_morest)

        if self.sampling_parameters['sampling_initialization']:
            self.log_morest.write('Start to sample the phase space\n\n')
            #Method: '+str(self.sampling_parameters['sampling_method'])+'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
            try:
                if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                    for md_file in glob('./MoREST_MD*'):
                        os.remove(md_file)
                elif self.sampling_parameters['sampling_method'].upper() in ['RPMD', 'RPMD_NM']:
                    tmp_file_list = ['./MoREST_RPMD.log', './MoREST_RPMD_traj.xyz']
                    for tmp_file in glob('./MoREST_RPMD_beads_traj_*'):
                        tmp_file_list.append(tmp_file)
                    for rpmd_file in tmp_file_list:
                        os.remove(rpmd_file)
            except:
                pass
        else:
            self.log_morest.write('Continue to sample the phase space\n\n')
            #Method: '+str(self.sampling_parameters['sampling_method'])+'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
        
        if not self.morest_parameters['enhanced_sampling']:
            self.sampling_job = self.generate_sampling_job(calculator=self.calculator, log_morest=self.log_morest)
        
    def generate_sampling_job(self, molecule=None, log_file_name=None, traj_file_name=None, T_simulation=None, calculator=None, log_morest=None):
        if self.sampling_parameters['sampling_method'].upper() in ['MD']:
            if self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_SVR']:
                sampling_job = NVT_SVR(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_SVR']:
                sampling_job = NPT_SVR(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                sampling_job = NVE_VV(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper() in ['NPH_SVR']:
                sampling_job = NPH_SVR(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Langevin'.upper()]:
                sampling_job = NVT_Langevin(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Langevin'.upper()]:
                sampling_job = NPT_Langevin(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Berendsen'.upper()]:
                sampling_job = NVT_Berendsen(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Berendsen'.upper()]:
                sampling_job = NPT_Berendsen(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVK_VR']:
                sampling_job = NVK_VR(self.morest_parameters, self.sampling_parameters, self.MD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            else:
                self.log_morest.write('It is not clear which ensemble will be used.\n')
                self.log_morest.close()
                raise Exception('Which ensemble will you use?')
        elif self.sampling_parameters['sampling_method'].upper() in ['RPMD']:
            if self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_SVR']:
                sampling_job = RP_NVT_SVR(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                sampling_job = RP_NVE_VV(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Langevin'.upper()]:
                sampling_job = RP_NVT_Langevin(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Berendsen'.upper()]:
                sampling_job = RP_NVT_Berendsen(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVK_VR']:
                sampling_job = RP_NVK_VR(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Langevin'.upper(), 'NPT_Berendsen'.upper(), 'NPH_SVR', 'NPT_SVR']:
                self.log_morest.write('The RPMD sampling method only supports NVE and NVT ensemble.\n')
                self.log_morest.close()
                raise Exception('The RPMD sampling method only supports NVE and NVT ensemble.')
            else:
                self.log_morest.write('It is not clear which ensemble will be used.\n')
                self.log_morest.close()
                raise Exception('Which ensemble will you use?')
        elif self.sampling_parameters['sampling_method'].upper() in ['RPMD_NM']:
            if self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_SVR']:
                sampling_job = RP_NVT_SVR_normal_mode(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                sampling_job = RP_NVE_VV_normal_mode(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Langevin'.upper()]:
                sampling_job = RP_NVT_Langevin_normal_mode(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Berendsen'.upper()]:
                sampling_job = RP_NVT_Berendsen_normal_mode(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVK_VR']:
                sampling_job = RP_NVK_VR_normal_mode(self.morest_parameters, self.sampling_parameters, self.RPMD_parameters, molecule, log_file_name, traj_file_name, T_simulation, calculator, log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Langevin'.upper(), 'NPT_Berendsen'.upper(), 'NPH_SVR', 'NPT_SVR']:
                self.log_morest.write('The normal mode sampling method only supports NVE and NVT ensemble.\n')
                self.log_morest.close()
                raise Exception('The normal mode sampling method only supports NVE and NVT ensemble.')
            else:
                self.log_morest.write('It is not clear which ensemble will be used.\n')
                self.log_morest.close()
                raise Exception('Which ensemble will you use?')
        else:
            self.log_morest.write('It is not clear which sampling method will be used.\n')
            self.log_morest.close()
            raise Exception('Will you use the phase sampling method?')
        return sampling_job

    def initialize_trajectory_scattering(self, MoREST_parameters):
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
                os.remove('MoREST_scattering.log')
                os.remove('MoREST_scattering.xyz')
                for scattering_file in glob('./MoREST_scattering_traj_*.log'):
                    os.remove(scattering_file)
                for scattering_file in glob('./MoREST_scattering_traj_*.xyz'):
                    os.remove(scattering_file)
            except:
                pass
        else:
            self.log_morest.write('Continue to sample the trajectories\n\n')

        if not self.morest_parameters['enhanced_sampling']:
            self.scattering_job = self.generate_scattering_job(calculator=self.calculator, log_morest=self.log_morest)
        
    def generate_scattering_job(self, calculator=None, log_morest=None):
        if self.scattering_parameters['scattering_method'].upper() in ['VV']:
            scattering_job = scattering_velocity_Verlet(self.morest_parameters, self.scattering_parameters, calculator, log_morest)
        elif self.scattering_parameters['scattering_method'].upper() in ['SY4']:
            scattering_job = scattering_Suzuki_Yoshida_4th(self.morest_parameters, self.scattering_parameters, calculator, log_morest)
        elif self.scattering_parameters['scattering_method'].upper() in ['RK4']:
            scattering_job = scattering_Runge_Kutta_4th(self.morest_parameters, self.scattering_parameters, calculator, log_morest)
        elif self.scattering_parameters['scattering_method'].upper() in ['LD']:
            scattering_job = scattering_Langevin_dynamics(self.morest_parameters, self.scattering_parameters, calculator, log_morest)
        else:
            self.log_morest.write('It is not clear which scattering method will be used.\n')
            self.log_morest.close()
            raise Exception('Which scattering method will you use?')
        return scattering_job

    def initialize_molecule_rovibrating(self, MoREST_parameters):
        if not self.morest_parameters['morest_load_parameters_file']:
            self.rovibrating_parameters = MoREST_parameters.get_rovibrating_parameters(self.log_morest)
        else:
            try:
                self.rovibrating_parameters = np.load('MoREST_rovibrating_parameters.npy',allow_pickle=True).item()
            except:
                self.log_morest.write('Can not find parameters files: MoREST_rovibrating_parameters.npy\n Read parameters from input file.\n\n')
                self.rovibrating_parameters = MoREST_parameters.get_rovibrating_parameters(self.log_morest)
        if self.rovibrating_parameters['rovibrating_initialization']:
            self.log_morest.write('Start to sample the rovibration states\n\n')
            try:
                os.remove('MoREST_rovibrating.log')
                os.remove('MoREST_rovibrating_traj.xyz')
            except:
                pass
        else:
            self.log_morest.write('Continue to sample the rovibration states\n\n')

        if self.rovibrating_parameters['rovibrating_method'].upper() in ['VV']:
            self.rovibrating_job = rovibrating_velocity_Verlet(self.morest_parameters, self.rovibrating_parameters, calculator=self.calculator, log_morest=self.log_morest)
        elif self.rovibrating_parameters['rovibrating_method'].upper() in ['SY4']:
            self.rovibrating_job = rovibrating_Suzuki_Yoshida_4th(self.morest_parameters, self.rovibrating_parameters, calculator=self.calculator, log_morest=self.log_morest)
        elif self.rovibrating_parameters['rovibrating_method'].upper() in ['RK4']:
            self.rovibrating_job = rovibrating_Runge_Kutta_4th(self.morest_parameters, self.rovibrating_parameters, calculator=self.calculator, log_morest=self.log_morest)
        else:
            self.log_morest.write('It is not clear which rovibrating method will be used.\n')
            self.log_morest.close()
            raise Exception('Which rovibrating method will you use?')

    def initialize_structure_searching(self, MoREST_parameters):
        if not self.morest_parameters['morest_load_parameters_file']:
            self.searching_parameters = MoREST_parameters.get_searching_parameters(self.log_morest)
            if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS','L-BFGS','L-BFGS-B','BFGS-TS','L-BFGS-TS','SGD', 'ADAM']:
                self.gradient_parameters = MoREST_parameters.get_gradient_parameters(self.log_morest)
            elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                self.fire_parameters = MoREST_parameters.get_fire_parameters(self.log_morest)
            elif self.searching_parameters['searching_method'].upper() in ['DIMER']:
                self.dimer_parameters = MoREST_parameters.get_dimer_parameters(self.log_morest)
            elif self.searching_parameters['searching_method'].upper() in ['GAD']:
                self.GAD_parameters = MoREST_parameters.get_GAD_parameters(self.log_morest)
        else:
            try:
                self.searching_parameters = np.load('MoREST_searching_parameters.npy',allow_pickle=True).item()
                if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS','L-BFGS','L-BFGS-B','BFGS-TS','L-BFGS-TS','SGD', 'ADAM']:
                    self.gradient_parameters = np.load('MoREST_gradient_parameters.npy',allow_pickle=True).item()
                elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                    self.fire_parameters = np.load('MoREST_FIRE_parameters.npy',allow_pickle=True).item()
                elif self.searching_parameters['searching_method'].upper() in ['DIMER']:
                    self.dimer_parameters = np.load('MoREST_dimer_parameters.npy',allow_pickle=True).item()
                elif self.searching_parameters['searching_method'].upper() in ['GAD']:
                    self.GAD_parameters = np.load('MoREST_GAD_parameters.npy',allow_pickle=True).item()
            except:
                self.log_morest.write('Can not find parameters files: MoREST_searching_parameters.npy, MoREST_FIRE_parameters.npy\n Read parameters from input file.\n\n')
                self.searching_parameters = MoREST_parameters.get_searching_parameters(self.log_morest)
                if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS','L-BFGS','L-BFGS-B','BFGS-TS','L-BFGS-TS','SGD', 'ADAM']:
                    self.gradient_parameters = MoREST_parameters.get_gradient_parameters(self.log_morest)
                elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                    self.fire_parameters = MoREST_parameters.get_fire_parameters(self.log_morest)
                elif self.searching_parameters['searching_method'].upper() in ['DIMER']:
                    self.dimer_parameters = MoREST_parameters.get_dimer_parameters(self.log_morest)
                elif self.searching_parameters['searching_method'].upper() in ['GAD']:
                    self.GAD_parameters = MoREST_parameters.get_GAD_parameters(self.log_morest)

        if self.searching_parameters['searching_initialization']:
            self.log_morest.write('Start to search the stationary structure.\n\n')
            #Method: '+str(self.sampling_parameters['sampling_method'])+'\nEnsemble: '+str(self.sampling_parameters['sampling_ensemble'])+'\n\n')
            try:
                #os.remove('MoREST.str_new')
                os.remove('MoREST_traj.xyz')
                os.remove('MoREST_'+self.searching_parameters['searching_method'].upper()+'.log')
            except:
                pass
        else:
            self.log_morest.write('Continue to search the stationary structure.\n\n')
        if not self.morest_parameters['enhanced_sampling']:
            self.searching_job = self.generate_searching_job(calculator=self.calculator, log_morest=self.log_morest)

    def generate_searching_job(self, molecule=None, log_file_name=None, traj_file_name=None, calculator=None, log_morest=None):
        if self.searching_parameters['searching_method'].upper() in ['GD', 'CG', 'BFGS', 'SGD', 'ADAM']:
            searching_job = gradient_descent(self.morest_parameters, self.searching_parameters, self.gradient_parameters, self.searching_parameters['searching_method'], \
                                             molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['L-BFGS']:
            searching_job = L_BFGS_descent(self.morest_parameters, self.searching_parameters, self.gradient_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['L-BFGS-B']:
            searching_job = scipy_L_BFGS_B_descent(self.morest_parameters, self.searching_parameters, self.gradient_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
            searching_job = FIRE_velocity_Verlet(self.morest_parameters, self.searching_parameters, self.fire_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['BFGS-TS']:
            searching_job = BFGS_TS(self.morest_parameters, self.searching_parameters, self.gradient_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['L-BFGS-TS']:
            searching_job = L_BFGS_TS(self.morest_parameters, self.searching_parameters, self.gradient_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['DIMER']:
            searching_job = dimer(self.morest_parameters, self.searching_parameters, self.dimer_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['GAD']:
            searching_job = GAD_velocity_Verlet(self.morest_parameters, self.searching_parameters, self.GAD_parameters, molecule, log_file_name, traj_file_name, calculator, log_morest)
        else:
            self.log_morest.write('It is not clear which searching method will be used.\n')
            self.log_morest.close()
            raise Exception('Will you use the structure searching method?')
        return searching_job

    def initialize_enhanced_sampling(self, MoREST_parameters):
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
        if self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['RE']:
            if not self.morest_parameters['phase_space_sampling']:
                self.log_morest.write('Replica exchange method needs phase space sampling.\n')
                raise Exception('Replica exchange method needs phase space sampling.')
            if not self.morest_parameters['morest_load_parameters_file']:
                self.re_parameters = MoREST_parameters.get_RE_parameters(self.log_morest)
            else:
                try:
                    self.re_parameters = np.load('MoREST_RE_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_RE_parameters.npy\n Read parameters from input file.\n\n')
                    self.re_parameters = MoREST_parameters.get_RE_parameters(self.log_morest)

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
            self.re_sampling = replica_exchange(self.re_parameters)
            molecules = self.re_sampling.get_current_molecules()
            if len(molecules) != len(self.re_parameters['re_replica_temperatures']):
                self.log_morest.write('The number of structures do not fit the number of temperatures.\n\n')
                raise Exception('The number of structures do not fit the number of temperatures.')
            log_file_name = self.re_sampling.get_log_file_name()
            traj_file_name = self.re_sampling.get_traj_file_name()
            self.sampling_job = []
            for i,T in enumerate(self.re_parameters['re_replica_temperatures']):
                tmp_sampling_job = self.generate_sampling_job(molecules[i], log_file_name[i], traj_file_name[i], T, self.calculator, self.log_morest)
                self.sampling_job.append(tmp_sampling_job)
                self.log_morest.write('Replica '+str(i)+' at '+str(T)+' K is ready.\n\n')
            self.log_morest.write('\n')
                
        elif self.enhanced_sampling_parameters['enhanced_sampling_method'].upper() in ['ITS']:
            if not self.morest_parameters['phase_space_sampling']:
                self.log_morest.write('Integrated tempering sampling method needs phase space sampling.\n')
                raise Exception('Integrated tempering sampling method needs phase space sampling.')
            if not self.morest_parameters['morest_load_parameters_file']:
                self.its_parameters = MoREST_parameters.get_ITS_parameters(self.log_morest)
            else:
                try:
                    self.its_parameters = np.load('MoREST_ITS_parameters.npy',allow_pickle=True).item()
                except:
                    self.log_morest.write('Can not find parameters files: MoREST_ITS_parameters.npy\n Read parameters from input file.\n\n')
                    self.its_parameters = MoREST_parameters.get_ITS_parameters(self.log_morest)
            if self.its_parameters['its_initialization']:
                try:
                    os.remove('MoREST_ITS_pk.npy')
                    os.remove('MoREST_ITS_nk.npy')
                    os.remove('MoREST_ITS_potential_energy.npy')
                except:
                    pass
                self.log_morest.write('Integrated tempering sampling method is initialized.\n\n')
            self.its_sampling = integrated_tempering_sampling(self.its_parameters)
        else:
            self.log_morest.write('It is not clear which enhanced sampling method will be used.\n')
            self.log_morest.close()
            raise Exception('Which enhanced sampling method will you use?')
        #for key in self.its_parameters:
        #    print(key+' : '+str(self.its_parameters[key]))

    def initialize_wall_potential(self, MoREST_parameters):
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
