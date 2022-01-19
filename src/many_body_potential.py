import numpy as np
import joblib
from copy import deepcopy

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
        representation = []
        Al1 = Al2F2.get_positions()[0]
        F2 = Al2F2.get_positions()[1]
        Al3 = Al2F2.get_positions()[2]
        F4 = Al2F2.get_positions()[3]

        r_Al1_F2 = np.linalg.norm(Al1 - F2)
        r_Al3_F4 = np.linalg.norm(Al3 - F4)
        r_Al1_F4 = np.linalg.norm(Al1 - F4)
        r_Al3_F2 = np.linalg.norm(Al3 - F2)
        r_Al1_Al3 = np.linalg.norm(Al1 - Al3)
        r_F2_F4 = np.linalg.norm(F2 - F4)
        R = np.linalg.norm((Al1 + F2)/2 - (Al3 + F4)/2)

        representation.append(1.0/r_Al1_F2)
        representation.append(1.0/r_Al3_F4)
        representation.append(1.0/r_Al1_F4)
        representation.append(1.0/r_Al3_F2)
        representation.append(1.0/r_Al1_Al3)
        representation.append(1.0/r_F2_F4)
        representation.append(1.0/R)
        representation.append(np.exp(-r_Al1_F2))
        representation.append(np.exp(-r_Al3_F4))
        representation.append(np.exp(-r_Al1_F4))
        representation.append(np.exp(-r_Al3_F2))
        representation.append(np.exp(-r_Al1_Al3))
        representation.append(np.exp(-r_F2_F4))
        representation.append(np.exp(-R))

        return np.array(representation)

    def get_ml_potential(self, system_list):
        if type(system_list) != list:
            raise ValueError
        representation_list = [self.generate_Al2F2_representation(i_system) for i_system in system_list]
        return self.ml_potential.predict(representation)
    
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
        energy_0 = energy_list[0]
        for i,i_energy in enumerate(energy_list[1:]):
            force_value = (i_energy - energy_0)/displacement
            forces.append(force_value)
        forces = np.array(forces)
        return energy_0, forces.reshape(n_atoms, 3)
            