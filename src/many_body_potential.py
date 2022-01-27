import numpy as np
import joblib
from copy import deepcopy
from ase import units

class ml_potential:
    '''
    This class implements loading machine learned many body potential and returning the potential as the output of the Cartesian coordinates input.
    INPUT:
    trained_ml_potential: trained model in scikit-learn format and loaded by joblib.
    #model_features: (numpy npy file) dictionary including the name of the features used in machine learning, loaded by numpy.
    #model_labels: (numpy npy file) dictionary including the name of the labels used in machine learning, loaded by numpy.
    system: ase.Atoms object
    
    '''
    def __init__(self, trained_ml_potential):
        self.ml_potential = joblib.load(trained_ml_potential)
        #self.ml_features = np.load(model_features, allow_pickle=True)
        #self.ml_labels = np.load(model_labels, allow_pickle=True)
        
    def generate_Al2F2_representation(self, Al2F2):
        
        Al1 = Al2F2.get_positions()[0] / units.Bohr # change length in AA to Bohr
        F2 = Al2F2.get_positions()[1] / units.Bohr
        Al3 = Al2F2.get_positions()[2] / units.Bohr
        F4 = Al2F2.get_positions()[3] / units.Bohr

        r_Al1_F2 = np.linalg.norm(Al1 - F2)
        r_Al3_F4 = np.linalg.norm(Al3 - F4)
        r_Al1_F4 = np.linalg.norm(Al1 - F4)
        r_Al3_F2 = np.linalg.norm(Al3 - F2)
        r_Al1_Al3 = np.linalg.norm(Al1 - Al3)
        r_F2_F4 = np.linalg.norm(F2 - F4)
        R = np.linalg.norm((Al1 + F2)/2 - (Al3 + F4)/2)

        inverse_r_Al1_F2 = 1.0/r_Al1_F2
        inverse_r_Al3_F4 = 1.0/r_Al3_F4
        inverse_r_Al1_F4 = 1.0/r_Al1_F4
        inverse_r_Al3_F2 = 1.0/r_Al3_F2
        inverse_r_Al1_Al3 = 1.0/r_Al1_Al3
        inverse_r_F2_F4 = 1.0/r_F2_F4
        inverse_R = 1.0/R
        exp_r_Al1_F2 = np.exp(-r_Al1_F2)
        exp_r_Al3_F4 = np.exp(-r_Al3_F4)
        exp_r_Al1_F4 = np.exp(-r_Al1_F4)
        exp_r_Al3_F2 = np.exp(-r_Al3_F2)
        exp_r_Al1_Al3 = np.exp(-r_Al1_Al3)
        exp_r_F2_F4 = np.exp(-r_F2_F4)
        exp_R = np.exp(-R)

        features_invr_AlF = np.array([inverse_r_Al1_F2,inverse_r_Al3_F4, inverse_r_Al1_F4, inverse_r_Al3_F2])
        features_invr_Al2_F2 = np.array([inverse_r_Al1_Al3, inverse_r_F2_F4, inverse_R])
        features_expr_AlF = np.array([exp_r_Al1_F2,exp_r_Al3_F4, exp_r_Al1_F4, exp_r_Al3_F2])
        features_expr_Al2_F2 = np.array([exp_r_Al1_Al3,exp_r_F2_F4, exp_R])

        representation = np.concatenate((np.sort(features_invr_AlF),
                                  features_invr_Al2_F2,
                                  np.sort(features_expr_AlF),
                                  features_expr_Al2_F2))

        return np.array(representation)

    def get_ml_potential(self, system_list):
        if type(system_list) != list:
            raise ValueError
        representation_list = [self.generate_Al2F2_representation(i_system) for i_system in system_list]
        return self.ml_potential.predict(representation_list) * units.Hartree # change energy in Hartree to eV
    
    def get_potential_FD_forces(self, system, displacement=0.0025):
        system_list = [system]
        n_atoms = system.get_number_of_atoms()
        forces = []
        for i in range(n_atoms):
            for j in range(3):
                new_system = deepcopy(system)
                coordinates = new_system.get_positions()
                coordinates[i,j] = coordinates[i,j] + displacement
                new_system.set_positions(coordinates)
                system_list.append(new_system)
        energy_list = self.get_ml_potential(system_list)
        #print(energy_list)
        energy_0 = energy_list[0]
        for i,i_energy in enumerate(energy_list[1:]):
            force_value = -1*(i_energy - energy_0)/displacement
            forces.append(force_value)
        forces = np.array(forces)
        #print(forces)
        return energy_0, forces.reshape(n_atoms, 3)
            