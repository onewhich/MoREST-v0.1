import numpy as np
#import pandas as pd
import pickle
from copy import deepcopy
from ase import units
from MoREST.structure_io import read_xyz_traj, write_xyz_traj
import subprocess
import os
from MoREST.collective_variable import collective_variables
from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.atoms import Atoms
from MoREAT.representation import generate_representation
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class on_the_fly:
    '''
    Using ase.calculators to calculates the potential energy and forces of the system during the simulation.
    INPUT:
    calculator: ase.calculators object that pass calculation parameters here
    '''
    def __init__(self, calculator):
        self.calculator = calculator
        
    def get_potential_forces(self, system):
        system.calc = self.calculator
        self.forces = system.get_forces()
        self.potential_energy = system.get_potential_energy()
        
        return self.potential_energy, self.forces

class ml_potential(Calculator):
    '''
    This class implements loading machine learned many body potential and returning the potential as the output of the Cartesian coordinates input.
    INPUT:
    trained_ml_potential: trained model in scikit-learn format and loaded by pickle.
    #model_features: (numpy npy file) dictionary including the name of the features used in machine learning, loaded by numpy.
    #model_labels: (numpy npy file) dictionary including the name of the labels used in machine learning, loaded by numpy.
    system: ase.Atoms object
    '''

    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    def __init__(self, restart=None, ignore_bad_restart=False, label='ml_potential', atoms=None, command=None, **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart=ignore_bad_restart, label=label, atoms=atoms, command=command, **kwargs)

        self.current_step = None
        self.log_morest = kwargs['log_file']
        self.if_print_uncertainty = kwargs['ml_parameters']['ml_print_uncertainty']
        self.if_fd_forces = kwargs['ml_parameters']['ml_fd_forces']
        if self.if_fd_forces:
            self.fd_displacement = kwargs['ml_parameters']['fd_displacement']
        self.if_active_learning = kwargs['ml_parameters']['ml_active_learning']
        if kwargs['ml_parameters']['ml_additional_features'] == None:
            self.additional_features = None
        else:
            self.additional_features = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features'])
        if kwargs['ml_parameters']['ml_additional_features_min'] == None:
            self.additional_features_min = None
        else:
            self.additional_features_min = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features_min'])
        if kwargs['ml_parameters']['ml_additional_features_max'] == None:
            self.additional_features_max = None
        else:
            self.additional_features_max = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features_max'])
        if self.if_active_learning:
            ab_initio_calculator = kwargs['ab_initio_calculator']
            if  ab_initio_calculator == None:
                raise Exception('Active learning is supposed to be used, please specify the electronic structure method.')
            try:
                tmp_noise_level_bounds = kwargs['ml_parameters']['ml_gpr_noise_level_bounds']
                if type(tmp_noise_level_bounds) == float:
                    self.noise_level_bounds = np.array([tmp_noise_level_bounds, 1.])
                elif type(tmp_noise_level_bounds) == np.ndarray:
                    self.noise_level_bounds = np.array([tmp_noise_level_bounds[0], tmp_noise_level_bounds[1]])
            except:
                self.noise_level_bounds = np.array([1e-7, 1.])
            try:
                self.filename_training_set = kwargs['ml_parameters']['ml_training_set']
                self.training_set = read_xyz_traj(self.filename_training_set)
            except:
                self.filename_training_set = 'training_set.xyz'
                self.training_set = []
            try:
                self.trained_ml_potential = kwargs['ml_parameters']['ml_potential_model']
                self.ml_potential = pickle.load(open(self.trained_ml_potential, 'rb'))
            except:
                self.log_morest.write('Trained ML model has not beed indicated. The ML model will be trained from training set.\n')
                self.ml_potential = self.train_ml_potential()
            self.energy_uncertainty_tolerance = kwargs['ml_parameters']['ml_energy_uncertainty_tolerance']
            self.appending_set_number = kwargs['ml_parameters']['ml_appending_set_number']
            self.appending_set_counter = 0
            if type(ab_initio_calculator) == type({}):
                molpro_para_dict = ab_initio_calculator
                self.ab_initio_potential = molpro_calculator(molpro_para_dict)
            else:
                self.ab_initio_potential = on_the_fly(ab_initio_calculator)
        else:
            try:
                self.trained_ml_potential = kwargs['ml_parameters']['ml_potential_model']
                self.ml_potential = pickle.load(open(self.trained_ml_potential, 'rb'))
            except:
                if 'ml_training_set' in kwargs['ml_parameters']:
                    self.training_set = read_xyz_traj(kwargs['ml_parameters']['ml_training_set'])
                    if 'ml_gpr_noise_level_bounds' in kwargs['ml_parameters']:
                        tmp_noise_level_bounds = kwargs['ml_parameters']['ml_gpr_noise_level_bounds']
                        if type(tmp_noise_level_bounds) == float:
                            self.noise_level_bounds = np.array([tmp_noise_level_bounds, 1.])
                        elif type(tmp_noise_level_bounds) == np.ndarray:
                            self.noise_level_bounds = np.array([tmp_noise_level_bounds[0], tmp_noise_level_bounds[1]])
                    else:
                        self.noise_level_bounds = np.array([1e-7, 1.])
                    self.log_morest.write('Trained ML model has not beed indicated. The ML model will be trained from training set.\n')
                    self.ml_potential = self.train_ml_potential()
                else:
                    raise Exception('ML model or training set can not be read. Please specify the name.')

    def calculate(self, *args, **kwargs):
        Calculator.calculate(self, *args, **kwargs)
        self.results['energy'], self.results['forces'] = self.get_potential_forces(self.atoms)

    def get_ml_potential(self, system_list):
        #if type(system_list) != list:
        #    raise ValueError
        #representation_list = [generate_representation.generate_Al2F2_representation(i_system) for i_system in system_list]
        representation_list = generate_representation(system_list).inverse_r_exp_r()
        if self.additional_features == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features.generate_CVs_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
        if self.additional_features_min == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features_min.generate_CV_min_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
        if self.additional_features_max == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features_max.generate_CV_max_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
        if self.if_fd_forces:
            ml_energy, ml_energy_std = self.ml_potential.predict(representation_list, return_std=True)
            ml_energy = np.array(ml_energy)
            ml_energy_std = np.array(ml_energy_std)
            return ml_energy, ml_energy_std
        else:
            ml_energy_forces, ml_energy_forces_std = self.ml_potential.predict(representation_list, return_std=True)
            ml_energy = np.array(ml_energy_forces[:,0])
            ml_forces = np.array(ml_energy_forces[:,1:]).reshape(-1,3)
            ml_energy_std = np.array(ml_energy_forces_std[:,0])
            ml_forces_std = np.array(ml_energy_forces_std[:,1:]).reshape(-1,3)
            return ml_energy, ml_energy_std, ml_forces, ml_forces_std

    def get_potential_forces(self, system):
        if type(system) == list:
            system = system[0]
        if self.if_fd_forces:
            system_list = [system]
            n_atoms = system.get_global_number_of_atoms()
            forces = []
            for i in range(n_atoms):
                for j in range(3):
                    new_system = deepcopy(system)
                    coordinates = new_system.get_positions()
                    coordinates[i,j] = coordinates[i,j] + self.fd_displacement
                    new_system.set_positions(coordinates)
                    system_list.append(new_system)
            # Get the predictions of energy and uncertainty
            energy_list, energy_std_list = self.get_ml_potential(system_list)
            #print("Energy:", energy_list)
            #print("Energy std:", energy_std_list)
            if len(np.shape(energy_list)) == 2:
                energy_list = energy_list.flatten()
            energy_0 = energy_list[0]
            energy_std_0 = energy_std_list[0]
            # Determine if the energy need to be calculated on the fly
            if self.if_active_learning and (energy_std_0 > self.energy_uncertainty_tolerance):
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy uncertainty is larger than tolerance(="+str(self.energy_uncertainty_tolerance)+"): "+str(energy_std_0)+"\n")
                self.log_morest.write("The relevant ML predicted potential energy: "+str(energy_0)+"\n")
                self.log_morest.write("Current system:\n")
                chemical_symbols = system.get_chemical_symbols()
                coordinates = system.get_positions()
                for i in range(len(coordinates)):
                    self.log_morest.write(chemical_symbols[i]+" "+str(coordinates[i][0])+" "+str(coordinates[i][1])+" "+str(coordinates[i][2])+"\n")
                #return float('nan'), float('nan')
                # If the ML energy has too large uncertainty, call ab initio calculations
                self.potential_energy, self.forces = self.ab_initio_potential.get_potential_forces(system)
                self.log_morest.write("The relevant ab initio potential energy: "+str(self.potential_energy)+"\n")
                write_xyz_traj(self.filename_training_set, system)
                self.training_set = read_xyz_traj(self.filename_training_set)
                self.log_morest.write("The current system has been added to the training set.\n")
                self.appending_set_counter += 1
                self.log_morest.write("The appending_set_counter is now: "+str(self.appending_set_counter)+"\n\n")
                if self.appending_set_counter >= self.appending_set_number:
                    self.log_morest.write("Start to train a new model:\n")
                    self.ml_potential = self.train_ml_potential()
                    self.appending_set_counter = 0
            else:
                if self.if_print_uncertainty:
                    self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                    self.log_morest.write("Current ML energy uncertainty: "+str(energy_std_0)+"\n\n")
                for i,i_energy in enumerate(energy_list[1:]):
                    force_value = -1*(i_energy - energy_0)/self.fd_displacement
                    forces.append(force_value)
                forces = np.array(forces)
                self.potential_energy = energy_0
                self.forces = forces.reshape(n_atoms, 3)
                #print('Predicted energy: ',energy_0)
                #print('Std error of the predicted energy: ',energy_std_0)
                #print('\n')
        else:
            potential_energy, potential_energy_std, forces, forces_std = self.get_ml_potential(system)
            #TODO: the RMSE of forces prediction is not used for judgment
            if self.if_active_learning and (potential_energy_std > self.energy_uncertainty_tolerance):
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy uncertainty is larger than tolerance(="+str(self.energy_uncertainty_tolerance)+"): "+str(potential_energy_std)+"\n")
                self.log_morest.write("The relevant ML predicted potential energy: "+str(potential_energy)+"\n")
                self.log_morest.write("Current ML forces uncertainty is: "+str(forces_std.flatten())+"\n")
                self.log_morest.write("The relevant ML predicted forces: "+str(forces.flatten())+"\n")
                self.log_morest.write("Current system:\n")
                chemical_symbols = system.get_chemical_symbols()
                coordinates = system.get_positions()
                for i in range(len(coordinates)):
                    self.log_morest.write(chemical_symbols[i]+" "+str(coordinates[i][0])+" "+str(coordinates[i][1])+" "+str(coordinates[i][2])+"\n")
                # If the ML energy has too large uncertainty, call ab initio calculations
                self.potential_energy, self.forces = self.ab_initio_potential.get_potential_forces(system)
                self.log_morest.write("The relevant ab initio potential energy: "+str(self.potential_energy)+"\n")
                write_xyz_traj(self.filename_training_set, system)
                self.training_set = read_xyz_traj(self.filename_training_set)
                self.log_morest.write("The current system has been added to the training set.\n")
                self.appending_set_counter += 1
                self.log_morest.write("The appending_set_counter is now: "+str(self.appending_set_counter)+"\n\n")
                if self.appending_set_counter >= self.appending_set_number:
                    self.log_morest.write("Start to train a new model:\n")
                    self.ml_potential = self.train_ml_potential()
                    self.appending_set_counter = 0
            else:
                if self.if_print_uncertainty:
                    self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                    self.log_morest.write("Current ML energy uncertainty: "+str(potential_energy_std)+"\n\n")
                self.potential_energy = potential_energy
                self.forces = forces
        return self.potential_energy, self.forces

    def get_appending_set_counter(self):
        return self.appending_set_counter

    def set_appending_set_counter(self, counter):
        self.appending_set_counter = counter

    def set_current_step(self, current_step):
        self.current_step = current_step

    @staticmethod
    def RMSE(true, pred):
        true = true.flatten()
        pred = pred.flatten()
        RMSE_value = 0.0
        N = 0
        #print(true, len(true))
        for i in range(len(true)):
            if(np.isnan(true[i])):
                continue
            if(np.isnan(pred[i])):
                continue
            RMSE_value += (pred[i] - true[i]) ** 2
            N += 1
        RMSE_value = np.sqrt(RMSE_value/N)
        return RMSE_value

    def train_ml_potential(self):
        """system_list: The trajectory for training set"""
        #self.log_morest.write("Model is training.\n")
        if len(self.training_set) < 1:
            raise Exception('The training set has no system.')
        x_train = generate_representation(self.training_set).inverse_r_exp_r()
        if self.additional_features == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features.generate_CVs_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
        if self.additional_features_min == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features_min.generate_CV_min_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
        if self.additional_features_max == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features_max.generate_CV_max_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
        np.savetxt('training_set_representation',x_train)

        if self.if_fd_forces:
            potential_energy_list = np.array([i_system.get_potential_energy() for i_system in self.training_set])
            y_train = potential_energy_list
        else:
            potential_energy_list = np.array([i_system.get_potential_energy() for i_system in self.training_set])
            forces_list = np.array([i_system.get_forces().flatten() for i_system in self.training_set])
            y_train = np.hstack((potential_energy_list,forces_list))
        np.savetxt('training_set_label',y_train)

        gpr_kernel=kernels.Matern(nu=2.5)*kernels.DotProduct(sigma_0=10)  + kernels.WhiteKernel(noise_level=0.1, \
                                                                            noise_level_bounds=(self.noise_level_bounds[0],self.noise_level_bounds[1]))
        self.log_morest.write("Training set:\n\tShape of feature: "+str(np.shape(x_train))+"\n")
        self.log_morest.write("\tShape of label: "+str(np.shape(y_train))+"\n")
        gpr = GaussianProcessRegressor(kernel=gpr_kernel,normalize_y=True)
        gpr.fit(x_train, y_train)
        with open(self.trained_ml_potential,'wb') as trained_model_file:
            pickle.dump(gpr, trained_model_file, protocol=4)
        self.log_morest.write("The trained kernel: "+str(gpr.kernel_)+"\n")

        y_train_pred, y_train_pred_std = gpr.predict(x_train, return_std=True)
        self.log_morest.write("Training RMSE: "+str(self.RMSE(y_train, y_train_pred))+"\n")
        self.log_morest.write("Training uncertainty: "+str(np.average(y_train_pred_std))+"\n")
        self.log_morest.write("Median training uncertainty: "+str(np.median(y_train_pred_std))+"\n\n")

        return gpr

    '''
    def train_ml_model_energy(self, feature_keys, label_keys, data_train, data_valid, if_train_forces):

        """
        feature_keys: 
        label_keys: ["total_energy"]
        """
        x_train = data_train[feature_keys]
        y_train = data_train[label_keys]
        # Todo: figure out where to generate the features, and if we should save the features, or only the xyz positions in the csv file?
        # Maybe we can generate the features outside this function upon calling, and give both data_training_set and x_train, y_train to this function to be saved.

        #gpr_kernel=kernels.Matern(nu=2.5)  + kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8,1e5))
        gpr_kernel=kernels.Matern(nu=2.5)*kernels.DotProduct(sigma_0=10)  + kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(2e-7,1e5))
        print("Training set: \n    Shape of feature: ", np.shape(x_train))
        print("Size of validation set:", np.shape(y_valid))
        gpr = GaussianProcessRegressor(kernel=gpr_kernel,normalize_y=True)
        print("The trained kernel: %s" % gpr.kernel_)

        # Get the training scores
        y_train_pred, y_train_pred_std = gpr.predict(x_train, return_std=True)
        print("Training RMSE:", self.RMSE(y_train, y_train_pred))
        print("Training uncertainty:", np.average(y_train_pred_std))
        
        data_train_save = x_train.copy(deep=True)
        data_train_save["total_energy_true"] = y_train
        data_train_save["total_energy_pred"] = y_train_pred
        data_train_save["total_energy_pred_std"] = y_train_pred_std
        data_train_save.to_csv("data_train-training_size_" + str(len(y_train)).zfill(6) + ".csv", index=None)
        
        # Get the validation scores
        y_valid_pred, y_valid_pred_std = gpr.predict(x_valid, return_std=True)
        print("Validation RMSE:", self.RMSE(y_valid, y_valid_pred))
        print("Validation uncertainty:", np.average(y_valid_pred_std))

        data_valid_save = x_valid.copy(deep=True)
        data_valid_save["total_energy_true"] = y_valid
        data_valid_save["total_energy_pred"] = y_valid_pred
        data_valid_save["total_energy_pred_std"] = y_valid_pred_std
        data_valid_save.to_csv("data_valid-training_size_" + str(len(y_train)).zfill(6) + ".csv", index=None)

        return gpr
        '''

#class ml_interface(Calculator):
#    '''
#    Interface of ASE for ml_potential
#    '''
#
#    implemented_properties = ['energy', 'forces']
#    discard_results_on_any_change = True
#
#    def __init__(self, restart=None, ignore_bad_restart=False, label='ml_potential', atoms=None, command=None, **kwargs):
#        Calculator.__init__(self, restart=restart, ignore_bad_restart=ignore_bad_restart, label=label, atoms=atoms, command=command, **kwargs)
#        self.ml_potential = ml_potential(**kwargs)
#
#    def calculate(self, *args, **kwargs):
#        Calculator.calculate(self, *args, **kwargs)
#        self.results['energy'], self.results['forces'] = self.ml_potential.get_potential_forces(self.atoms)
#
#    def read(self, *args, **kwargs):
#        pass

class Molpro(FileIOCalculator):
    '''
    molpro calculator interface for ASE.
    This class can not be used directly by MoREST, but via on_the_fly many body potential method, while the calculator is redirected to this class.
    '''
    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    def __init__(self, *args, **kwargs):
        self.molpro_dir = kwargs['molpro_dir']
        self.ntasks = kwargs['ntasks']
        self.nthreads = kwargs['nthreads']
        self.method = kwargs['method']
        self.basis = kwargs['basis']
        self.geomtyp = 'xyz'
        try:
            self.overwrite = kwargs['overwrite']
            if self.overwrite.upper() in ['True'.upper()]:
                self.overwrite = True
            elif self.overwrite.upper() in ['False'.upper()]:
                self.overwrite = False
        except:
            self.overwrite = False
        try:
            self.memory = kwargs['memory']
        except:
            self.memory='12,g'
        try:
            self.unit = kwargs['unit']
        except:
            self.unit='angstrom'
        try:
            self.infile = kwargs['infile']
        except:
            self.infile='molpro.inp'
        try:
            self.outfile = kwargs['outfile']
        except:
            self.outfile='molpro.out'
        try:
            self.noforce = kwargs['noforce']
            if self.noforce.upper() in ['True'.upper()]:
                self.noforce = True
            elif self.noforce.upper() in ['False'.upper()]:
                self.noforce = False
        except:
            self.noforce = False
        FileIOCalculator.__init__(self, *args, **kwargs)

    def calculate(self,  *args, **kwargs):
        if self.overwrite:
            self.command = self.molpro_dir + " -n " + str(self.ntasks) + " -t " + str(self.nthreads) + r" -W $PWD/wfu "  + \
                            " -o " + self.outfile + " " + self.infile
        else:
            self.command = self.molpro_dir + " -n " + str(self.ntasks) + " -t " + str(self.nthreads) + r" -W $PWD/wfu "  + self.infile
        FileIOCalculator.calculate(self, *args, **kwargs)

    def write_input(self, system, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, system, properties, system_changes)
        self.n_atom = system.get_global_number_of_atoms()
        self.elements = system.get_chemical_symbols()
        self.positions = system.get_positions()
        inpstr = 'memory,'+self.memory+'\n\n'
        inpstr += 'symmetry,nosym\n\n'
        inpstr += self.unit + '\n\n'
        # Parse the geometry
        inpstr += 'geomtyp=xyz\n'
        inpstr += 'geometry={\n'+str(self.n_atom)+'\n\n'
        for i,element in enumerate(self.elements):
            inpstr += element
            inpstr += ' '
            for j in range(3):
                inpstr += ' ' + str(self.positions[i][j])
            inpstr += '\n'
        inpstr += '}'
        inpstr += '\n'
        # Parse the basis
        inpstr += '\nbasis=' + self.basis
        # Parse the method
        if not self.noforce:
            if not 'force' in self.method:
                inpstr += '\n' + self.method + '\nforce'
            else:
                inpstr += '\n' + self.method
        else:
            inpstr += '\n' + self.method
        # Write the input file
        with open(self.infile, 'w') as fin:
            fin.write(inpstr)

    def read_results(self, outfile=None):
        if outfile == None:
            outfile = self.outfile
        self.results['energy'], self.results['forces'] = self.parse_outfile(outfile)

    @staticmethod
    def parse_outfile(file):
        """
        Gets the coordinates and energies from molpro single-point calculation outputs (in Bohr)
        Returns:
            path: Path of molpro output file
            elements: Elements of the atoms  --> symbol = ''.join(elements); 
                                                ase.Atoms.set_chemical_symbols([elements]); 
            xyz: Coordinates of the atoms  --> ase.Atoms.set_positions([xyz])
            energy: Total energy of the system
            force (if_get_force = True): Force of atoms             
        Example:
            file = "molpro.out"
            path, elements, xyz, energy, force = get_xyz_energy(file, if_get_force=True)
        Sample Molpro outputs:
            - Geometry:

                ATOMIC COORDINATES

                NR  ATOM    CHARGE       X              Y              Z

                1  AL     13.00    0.000000000    2.362157656    0.000000000
                2  F       9.00    0.000000000   -2.362157656    0.000000000
                3  AL     13.00    0.000000000    8.031336029    0.000000000
                4  F       9.00    0.000000000    3.307020718    0.000000000
                
                Bond lengths in Bohr (Angstrom)

                1-3  5.669178374  1-4  0.944863062
                    ( 3.000000000)     ( 0.500000000)

                Bond angles

                3-1-4    0.00000000

            - Gradient:

                CCSD(T) GRADIENT FOR STATE 1.1

                Atom          dE/dx               dE/dy               dE/dz

                1         0.000000000        -0.000000000        -0.070539245
                2        -0.000000000         0.000000000         0.070539245

                Nuclear force contribution to virial =         0.266599708
        """

        with open(file,'r') as f:
            lines = f.readlines()
            energy = np.nan
            elements = []
            force = []
            if(len(lines)<1):
                energy = np.inf
                return energy, np.array(force)
            if(lines[-1].find("terminated")==-1):
                energy = np.inf
                return energy, np.array(force)
            for i, line in enumerate(lines):
                if(line.find("GRADIENT FOR STATE")!=-1):
                    ii = i+4
                    forcexs = []
                    forceys = []
                    forcezs = []
                    elements = []
                    while (lines[ii].find("Nuclear force contribution")==-1) and (len(lines[ii].split())>3):
                        element = lines[ii].split()[1]
                        x = float(lines[ii].split()[-3])
                        y = float(lines[ii].split()[-2])
                        z = float(lines[ii].split()[-1])
                        elements.append(element)
                        forcexs.append(x)
                        forceys.append(y)
                        forcezs.append(z)
                        ii += 1            
                    force = []
                    for i_atom in range(len(forcexs)):
                        force.append([forcexs[i_atom],forceys[i_atom],forcezs[i_atom]])
                """
                Energy:
                
                CCSD(T)/aug-cc-pVQZ energy=   -671.623485056226
                """
                if(line.find(" energy=")!=-1):
                    energy = float(line.split("=")[-1])
            return energy * units.Hartree, np.array(force) * (units.Hartree/units.Bohr) * -1

    @staticmethod
    def get_atoms_from_output(file):
        molpro_output = open(file,'r').readlines()
        for i,i_line in enumerate(molpro_output):
            if "geometry=" in i_line:
                n_atoms = int(molpro_output[i+1])
                chemical_symbols = []
                positions = []
                for i_atom in range(n_atoms+1):
                    i_check_line = i+2+i_atom
                    if len(molpro_output[i_check_line].split()) == 4:
                        element = molpro_output[i_check_line].split()[0]
                        chemical_symbols.append(element)
                        tmp_pos = np.array(molpro_output[i_check_line].split()[1:4],dtype=float)
                        positions.append(tmp_pos)
        return chemical_symbols, np.array(positions)

    @staticmethod
    def get_system_from_output(file):
        """
        Get the ase.Atoms object from the molpro output file.
        """
        chemical_symbols, positions = Molpro.get_atoms_from_output(file)
        energy, forces = Molpro.parse_outfile(file)
        system = Atoms(chemical_symbols, positions=positions)
        system.calc = SinglePointCalculator(system, energy=energy, forces=forces)
        return system

class molpro_calculator:
    '''
    molpro calculator for direct using.
    '''
    def __init__(self, molpro_para_dict):
        self.molpro_dir = molpro_para_dict['molpro_dir']
        self.ntasks = molpro_para_dict['ntasks']
        self.nthreads = molpro_para_dict['nthreads']
        self.method = molpro_para_dict['method']
        self.basis = molpro_para_dict['basis']
        self.geomtyp = 'xyz'
        try:
            self.overwrite = molpro_para_dict['overwrite']
        except:
            self.overwrite = False
        try:
            self.memory = molpro_para_dict['memory']
        except:
            self.memory='12,g'
        try:
            self.unit = molpro_para_dict['unit']
        except:
            self.unit='angstrom'
        try:
            self.infile = molpro_para_dict['infile']
        except:
            self.infile='molpro.inp'
        try:
            self.outfile = molpro_para_dict['outfile']
        except:
            self.outfile='molpro.out'


    def get_potential_forces(self, system):
        if os.path.isfile(self.outfile):
            self.potential_energy, self.forces = self.parse_outfile(self.outfile, if_get_force=True)
        else:
            self.n_atom = system.get_global_number_of_atoms()
            self.elements = system.get_chemical_symbols()
            self.positions = system.get_positions()
            self.run_molpro()
            self.potential_energy, self.forces = self.parse_outfile(self.outfile, if_get_force=True)
        return self.potential_energy, self.forces

    def run_molpro(self):
        #runcommand = self.molpro_dir + " < " + self.infile + " > " + self.outfile
        if self.overwrite:
            runcommand = self.molpro_dir + " -n " + str(self.ntasks) + " -t " + str(self.nthreads) + r" -W $PWD/wfu "  + \
                            " -o " + self.outfile + " " + self.infile
        else:
            runcommand = self.molpro_dir + " -n " + str(self.ntasks) + " -t " + str(self.nthreads) + r" -W $PWD/wfu "  + self.infile
        inpstr = 'memory,'+self.memory+'\n\n'
        inpstr += 'symmetry,nosym\n\n'
        inpstr += self.unit + '\n\n'
        # Parse the geometry
        inpstr += 'geomtyp=xyz\n'
        inpstr += 'geometry={\n'+str(self.n_atom)+'\n\n'
        for i,element in enumerate(self.elements):
            inpstr += element
            inpstr += ' '
            for j in range(3):
                inpstr += ' ' + str(self.positions[i][j])
            inpstr += '\n'
        inpstr += '}'
        inpstr += '\n'
        # Parse the basis
        inpstr += '\nbasis=' + self.basis
        # Parse the method
        if not 'force' in self.method:
            inpstr += '\n' + self.method + '\nforce'
        else:
            inpstr += '\n' + self.method
        # Write the input file
        with open(self.infile, 'w') as fin:
            fin.write(inpstr)
        #print(runcommand)
        runresult = subprocess.run(runcommand, shell=True)
        runresult = subprocess.run('rm -f *.xml', shell=True)
        #print("Molpro exit code:", runresult.returncode)
        return runresult.returncode
        #os.system(runcommand)
        #os.system('rm -f *.xml')

    @staticmethod
    def parse_outfile(file, if_get_force=True):
        """
        Gets the coordinates and energies from molpro single-point calculation outputs (in Bohr)
        Returns:
            path: Path of molpro output file
            elements: Elements of the atoms  --> symbol = ''.join(elements); 
                                                ase.Atoms.set_chemical_symbols([elements]); 
            xyz: Coordinates of the atoms  --> ase.Atoms.set_positions([xyz])
            energy: Total energy of the system
            force (if_get_force = True): Force of atoms             
        Example:
            file = "molpro.out"
            path, elements, xyz, energy, force = get_xyz_energy(file, if_get_force=True)
        Sample Molpro outputs:
            - Geometry:

                ATOMIC COORDINATES

                NR  ATOM    CHARGE       X              Y              Z

                1  AL     13.00    0.000000000    2.362157656    0.000000000
                2  F       9.00    0.000000000   -2.362157656    0.000000000
                3  AL     13.00    0.000000000    8.031336029    0.000000000
                4  F       9.00    0.000000000    3.307020718    0.000000000
                
                Bond lengths in Bohr (Angstrom)

                1-3  5.669178374  1-4  0.944863062
                    ( 3.000000000)     ( 0.500000000)

                Bond angles

                3-1-4    0.00000000

            - Gradient:

                CCSD(T) GRADIENT FOR STATE 1.1

                Atom          dE/dx               dE/dy               dE/dz

                1         0.000000000        -0.000000000        -0.070539245
                2        -0.000000000         0.000000000         0.070539245

                Nuclear force contribution to virial =         0.266599708
        """

        #unit_xyz = "Bohr"
        #unit_energy = "Hartree"
        #print("\n\nFile:", file)
        #path = ""
        with open(file,'r') as f:
            #path = file
            lines = f.readlines()
            energy = np.nan
            #xs = []
            #ys = []
            #zs = []
            elements = []
            #xyz = []
            force = []
            if(len(lines)<1):
                energy = np.inf
                #return path, elements, xyz_Al1F2Al3F4, energy
                if if_get_force:
                    return energy, np.array(force)
                else:
                    return energy
            if(lines[-1].find("terminated")==-1):
                energy = np.inf
                #return path, elements, xyz_Al1F2Al3F4, energy
                if if_get_force:
                    return energy, np.array(force)
                else:
                    return energy
            for i, line in enumerate(lines):
                if(line.find("GRADIENT FOR STATE")!=-1):
                    ii = i+4
                    forcexs = []
                    forceys = []
                    forcezs = []
                    elements = []
                    while (lines[ii].find("Nuclear force contribution")==-1) and (len(lines[ii].split())>3):
                        element = lines[ii].split()[1]
                        x = float(lines[ii].split()[-3])
                        y = float(lines[ii].split()[-2])
                        z = float(lines[ii].split()[-1])
                        elements.append(element)
                        forcexs.append(x)
                        forceys.append(y)
                        forcezs.append(z)
                        ii += 1            
                    force = []
                    for i_atom in range(len(forcexs)):
                        force.append([forcexs[i_atom],forceys[i_atom],forcezs[i_atom]])
            #for i, line in enumerate(lines):
            #    if(line.find("ATOMIC COORDINATES")!=-1):
            #        ii = i+4
            #        xs = []
            #        ys = []
            #        zs = []
            #        elements = []
            #        while (lines[ii].find("Bond lengths")==-1) and (len(lines[ii].split())>3):
            #            element = lines[ii].split()[1]
            #            x = float(lines[ii].split()[-3])
            #            y = float(lines[ii].split()[-2])
            #            z = float(lines[ii].split()[-1])
            #            elements.append(element)
            #            xs.append(x)
            #            ys.append(y)
            #            zs.append(z)
            #            ii += 1
                """
                Energy:
                
                CCSD(T)/aug-cc-pVQZ energy=   -671.623485056226
                """
                #if(line.find("Bond lengths in ")!= -1):
                #    print("Unit of coordinates read from the output file:", line.split("Bond lengths in ")[-1].split(" ")[0])
                if(line.find(" energy=")!=-1):
                    energy = float(line.split("=")[-1])
            
            #for i_atom in range(len(xs)):            
            #    xyz.append([xs[i_atom],ys[i_atom],zs[i_atom]])
            
            #print(" \nElements:", elements, "; Coordinates:", xyz, "; \n Energy:", energy, ";\n Force:", force)

            #if if_get_force:
            #    return path, elements, xyz, energy, force
            #else:
            #    return path, elements, xyz, energy

            if if_get_force:
                return energy * units.Hartree, np.array(force) * (units.Hartree/units.Bohr) * -1
            else:
                return energy * units.Hartree


