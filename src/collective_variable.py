from tabnanny import check
import numpy as np

class collective_variable:
    def __init__(self, from_CVs_file=False, CVs_list=None, CVs_file='MoREST.CVs'):
        if from_CVs_file:
            self.CVs_list = open(CVs_file,'r').readlines()
            for i, i_line in enumerate(self.CVs_list):
                self.CVs_list[i] = i_line.split()
        else:
            self.CVs_list = CVs_list
            #print(type(self.CVs_list))   # DEBUG
            #print(self.CVs_list)         # DEBUG
            if type(self.CVs_list) != type([]): # type(list) returns type
                raise Exception('Please specify the CVs in a list (CVs_list).')

    def generate_checks(self, system, output=False):
        checks = []
        for i_CV in self.CVs_list:
            if i_CV[0] == 'central_R_one':
                if i_CV[3] == 'all':
                    checks.append(self.central_R_one(system, i_CV[1], i_CV[2], i_CV[3]))
                else:
                    checks.append(self.central_R_one(system, i_CV[1], i_CV[2], i_CV[3], i_CV[4]))
            if i_CV[0] == 'central_R_all':
                if i_CV[3] == 'all':
                    checks.append(self.central_R_all(system, i_CV[1], i_CV[2], i_CV[3]))
                else:
                    checks.append(self.central_R_all(system, i_CV[1], i_CV[2], i_CV[3], i_CV[4]))
            if i_CV[0] == 'distance':
                checks.append(self.central_R_all(system, i_CV[1], i_CV[2], i_CV[3]))
        return np.array(checks)

    def check_CVs_one(self, system, output=False):
        checks = self.generate_checks(system, output)
        if output:
            pass
        else:
            if np.sum(np.where(checks == True, 1, 0)) > 0:
                return True
            else:
                return False

    def check_CVs_all(self, system, output=False):
        checks = self.generate_checks(system, output)
        if output:
            pass
        else:
            if np.sum(np.where(checks == False, 1, 0)) > 0:
                return False
            else:
                return True

    def central_R_one(self, system, checker, cutoff, N_check, atom_list=None):
        coordinates = system.get_positions()
        if N_check == 'all':
            central_R = np.linalg.norm(coordinates, axis=1)
        else:
            atom_list = np.array(atom_list)
            atom_list -= 1
            central_R = np.linalg.norm(coordinates[atom_list], axis=1)
        if checker == -1:
            check_result = np.where(central_R < cutoff, 1, 0)
            if np.sum(check_result) > 0:
                return True
            else:
                return False
        elif checker == 1:
            check_result = np.where(central_R > cutoff, 1, 0)
            if np.sum(check_result) > 0:
                return True
            else:
                return False

    def central_R_all(self, system, checker, cutoff, N_check, atom_list=None):
        coordinates = system.get_positions()
        if N_check == 'all':
            central_R = np.linalg.norm(coordinates, axis=1)
        else:
            atom_list = np.array(atom_list)
            atom_list -= 1
            central_R = np.linalg.norm(coordinates[atom_list], axis=1)
        if checker == -1:
            check_result = np.where(central_R < cutoff, 1, 0)
            if np.sum(check_result) == len(central_R):
                return True
            else:
                return False
        elif checker == 1:
            check_result = np.where(central_R > cutoff, 1, 0)
            if np.sum(check_result) == len(central_R):
                return True
            else:
                return False

    def distance(self, system, checker, cutoff, atom1, atom2):
        coordinates = system.get_positions()
        atom1_coordinate = coordinates[atom1-1]
        atom2_coordinate = coordinates[atom2-1]
        R_1_2 = np.linalg.norm(atom1_coordinate - atom2_coordinate)
        if checker == -1:
            if R_1_2 < cutoff:
                return True
            else:
                return False
        elif checker == 1:
            if R_1_2 > cutoff:
                return True
            else:
                return False
