import os
from glob import glob
import numpy as np
from many_body_potential import ml_potential, on_the_fly, molpro_calculator

class initialize_calculator:
    def __init__(self, morest_parameters, calculator=None, log_morest=None):
        self.morest_parameters = morest_parameters
        self.log_morest = log_morest
        
        if self.morest_parameters['many_body_potential'].upper() in ['on_the_fly'.upper()]:
            if calculator == None:
                raise Exception('Please specify the electronic structure method.')
            self.many_body_potential = on_the_fly(calculator)
            self.calculator = calculator
        elif self.morest_parameters['many_body_potential'].upper() in ['molpro'.upper()]:
            if type(calculator) == type({}):
                molpro_para_dict = calculator
                self.many_body_potential = molpro_calculator(molpro_para_dict)
            else:
                raise Exception('Please pass the molpro parameters dictionary to calculator.')
            self.calculator = calculator
        elif self.morest_parameters['many_body_potential'].upper() in ['ML_potential'.upper()]:
            self.ml_calculator = ml_potential(ab_initio_calculator = calculator, \
                                    ml_parameters = self.morest_parameters, \
                                    log_file = self.log_morest)
            self.many_body_potential = on_the_fly(self.ml_calculator)
            self.calculator = self.ml_calculator
            
        else:
            raise Exception('Which many body potential will you use?')
        
    def get_current_calculator(self):
        return self.calculator
    
from phase_space_sampling import NVE_VV, NVT_VR, NVT_Berendsen, NVT_Langevin, NVT_SVR, NPT_Berendsen, NPT_Langevin, NPT_BZP
from trajectory_scattering import scattering_velocity_Verlet, scattering_Runge_Kutta_4th
from structure_searching import gradient_descent, fire_velocity_Verlet
from enhanced_sampling import its, re
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
                self.md_parameters = MoREST_parameters.get_md_parameters(self.log_morest)
        else:
            try:
                self.sampling_parameters = np.load('MoREST_sampling_parameters.npy',allow_pickle=True).item()
                if self.sampling_parameters['sampling_method'].upper() in ['MD']:
                    self.md_parameters = np.load('MoREST_MD_parameters.npy',allow_pickle=True).item()
            except:
                self.log_morest.write('Can not find parameters files: MoREST_sampling_parameters.npy, MoREST_MD_parameters.npy\n Read parameters from input file.\n\n')
                self.sampling_parameters = MoREST_parameters.get_sampling_parameters(self.log_morest)
                if self.sampling_parameters['sampling_method'].upper() in ['MD']:
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
            if self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_SVR']:
                self.sampling_job = NVT_SVR(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_BZP']:
                self.sampling_job = NPT_BZP(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                self.sampling_job = NVE_VV(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Langevin'.upper()]:
                self.sampling_job = NVT_Langevin(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Langevin'.upper()]:
                self.sampling_job = NPT_Langevin(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Berendsen'.upper()]:
                self.sampling_job = NVT_Berendsen(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Berendsen'.upper()]:
                self.sampling_job = NPT_Berendsen(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_VR']:
                self.sampling_job = NVT_VR(self.morest_parameters, self.sampling_parameters, self.md_parameters, calculator=self.calculator, log_morest=self.log_morest)
            else:
                self.log_morest.write('It is not clear which ensemble will be used.\n')
                self.log_morest.close()
                raise Exception('Which ensemble will you use?')
        else:
            self.log_morest.write('It is not clear which sampling method will be used.\n')
            self.log_morest.close()
            raise Exception('Will you use the phase sampling method?')

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
                #os.remove('MoREST.str')
                for scattering_file in glob('./MoREST_traj_*.xyz'):
                    os.remove(scattering_file)
                for scattering_file in glob('./MoREST_traj_*.log'):
                    os.remove(scattering_file)
            except:
                pass
        else:
            self.log_morest.write('Continue to sample the trajectories\n\n')

        if self.scattering_parameters['scattering_method'].upper() in ['VV']:
            self.scattering_job = scattering_velocity_Verlet(self.morest_parameters, self.scattering_parameters, calculator=self.calculator, log_morest=self.log_morest)
        elif self.scattering_parameters['scattering_method'].upper() in ['RK4']:
            self.scattering_job = scattering_Runge_Kutta_4th(self.morest_parameters, self.scattering_parameters, calculator=self.calculator, log_morest=self.log_morest)
        else:
            self.log_morest.write('It is not clear which scattering method will be used.\n')
            self.log_morest.close()
            raise Exception('Which scattering method will you use?')

    def initialize_structure_searching(self, MoREST_parameters):
        if not self.morest_parameters['morest_load_parameters_file']:
            self.searching_parameters = MoREST_parameters.get_searching_parameters(self.log_morest)
            if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS']:
                self.gradient_parameters = MoREST_parameters.get_gradient_parameters(self.log_morest)
            elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                self.fire_parameters = MoREST_parameters.get_fire_parameters(self.log_morest)
        else:
            try:
                self.searching_parameters = np.load('MoREST_searching_parameters.npy',allow_pickle=True).item()
                if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS']:
                    self.gradient_parameters = np.load('MoREST_gradient_parameters.npy',allow_pickle=True).item()
                elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                    self.fire_parameters = np.load('MoREST_FIRE_parameters.npy',allow_pickle=True).item()
            except:
                self.log_morest.write('Can not find parameters files: MoREST_searching_parameters.npy, MoREST_FIRE_parameters.npy\n Read parameters from input file.\n\n')
                self.searching_parameters = MoREST_parameters.get_searching_parameters(self.log_morest)
                if self.searching_parameters['searching_method'].upper() in ['GD','CG','BFGS']:
                    self.gradient_parameters = MoREST_parameters.get_gradient_parameters(self.log_morest)
                elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
                    self.fire_parameters = MoREST_parameters.get_fire_parameters(self.log_morest)

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
        if self.searching_parameters['searching_method'].upper() in ['GD', 'CG', 'BFGS']:
            self.searching_job = gradient_descent(self.morest_parameters, self.searching_parameters, self.gradient_parameters, calculator=self.calculator, \
                                                   method=self.searching_parameters['searching_method'], log_morest=self.log_morest)
        elif self.searching_parameters['searching_method'].upper() in ['FIRE']:
            self.searching_job = fire_velocity_Verlet(self.morest_parameters, self.searching_parameters, self.fire_parameters, calculator=self.calculator, \
                                                      log_morest=self.log_morest)
        else:
            self.log_morest.write('It is not clear which searching method will be used.\n')
            self.log_morest.close()
            raise Exception('Will you use the structure searching method?')

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
                    if self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_SVR']:
                        tmp_sampling_job = NVT_SVR(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                    log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_BZP']:
                        tmp_sampling_job = NPT_BZP(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                    log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper() in ['NVE_VV']:
                        tmp_sampling_job = NVE_VV(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                   log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Langevin'.upper()]:
                        tmp_sampling_job = NVT_Langevin(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                         log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Langevin'.upper()]:
                        tmp_sampling_job = NPT_Langevin(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                         log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_Berendsen'.upper()]:
                        tmp_sampling_job = NVT_Berendsen(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                          log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NPT_Berendsen'.upper()]:
                        tmp_sampling_job = NPT_Berendsen(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                          log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    elif self.sampling_parameters['sampling_ensemble'].upper()  in ['NVT_VR']:
                        tmp_sampling_job = NVT_VR(self.morest_parameters, self.sampling_parameters, self.md_parameters, molecules[i], \
                                                   log_file_name[i], traj_file_name[i], T, calculator=self.calculator, log_morest=self.log_morest)
                    else:
                        self.log_morest.write('It is not clear which ensemble will be used.\n')
                        self.log_morest.close()
                        raise Exception('Which ensemble will you use?')
                self.sampling_job.append(tmp_sampling_job)
                self.log_morest.write('Replica '+str(i)+' at '+str(T)+' K is ready.\n\n')
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
