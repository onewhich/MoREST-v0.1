#from tabnanny import check
import numpy as np

class collective_variables:
    def __init__(self, from_CVs_file=False, CVs_list=None, CVs_file='MoREST.CVs'):
        if from_CVs_file: #TODO
            self.CVs = open(CVs_file,'r').readlines()
            self.CVs_list = []
            for i, i_line in enumerate(self.CVs):
                self.CVs_list.append(i_line.split())
        else:
            self.CVs_list = CVs_list
            #print(type(self.CVs_list))   # DEBUG
            #print(self.CVs_list)         # DEBUG
            if self.CVs_list == None:
                raise Exception('Please specify the CVs in a list (CVs_list).')
            elif type(self.CVs_list) != type([]): # type(list) returns type
                self.CVs_list = [CVs_list]

    def generate_checks(self, system, output=False):
        checks = []
        for i_CV in self.CVs_list:
            if i_CV[0] == 'central_R_one':
                checks.append(self.check_central_R_one(system, i_CV[1], i_CV[2], i_CV[3]))
            elif i_CV[0] == 'central_R_all':
                checks.append(self.check_central_R_all(system, i_CV[1], i_CV[2], i_CV[3]))
            elif i_CV[0] == 'distance':
                checks.append(self.check_distance(system, i_CV[1], i_CV[2], i_CV[3], i_CV[4]))
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

    def check_central_R_one(self, system, checker, cutoff, atom_list=None):
        central_R_list = self.central_R(system, atom_list)
        if checker == -1:
            check_result = np.where(central_R_list < cutoff, 1, 0)
            if np.sum(check_result) > 0:
                return True
            else:
                return False
        elif checker == 1:
            check_result = np.where(central_R_list > cutoff, 1, 0)
            if np.sum(check_result) > 0:
                return True
            else:
                return False

    def check_central_R_all(self, system, checker, cutoff, atom_list=None):
        central_R_list = self.central_R(system, atom_list)
        if checker == -1:
            check_result = np.where(central_R_list < cutoff, 1, 0)
            if np.sum(check_result) == len(central_R_list):
                return True
            else:
                return False
        elif checker == 1:
            check_result = np.where(central_R_list > cutoff, 1, 0)
            if np.sum(check_result) == len(central_R_list):
                return True
            else:
                return False

    def check_distance(self, system, checker, cutoff, atom1, atom2):
        R_1_2 = self.distance(system, atom1, atom2)
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

    def generate_CVs_list(self, system_list):
        if type(system_list) == type([]):
            return np.array([self.generate_collective_variables(i_sys) for i_sys in system_list])
        else:
            return np.array([self.generate_collective_variables(i_sys) for i_sys in [system_list]])

    def generate_collective_variables(self, system):
        CVs = []
        for i_CV in self.CVs_list:
            if i_CV[0] == 'inverse_r':
                CVs.append(self.inverse_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'exp_r':
                CVs.append(self.exp_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'inverse_r_exp_r':
                for i in self.inverse_r_exp_r(system, i_CV[1], i_CV[2]):
                    CVs.append(i)
            elif i_CV[0] == 'distance':
                CVs.append(self.distance(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'central_R':
                for i in self.central_R(system, i_CV[1]):
                    CVs.append(i)
            elif i_CV[0] == 'min_distance':
                CVs.append(self.min_distance(system, i_CV[1]))
            elif i_CV[0] == 'max_distance':
                CVs.append(self.max_distance(system, i_CV[1]))
        return np.array(CVs)
    
    def generate_CV_min_list(self, system_list):
        if type(system_list) == type([]):
            return np.array([self.generate_CV_min(i_sys) for i_sys in system_list])
        else:
            return np.array([self.generate_CV_min(i_sys) for i_sys in [system_list]])
        
    def generate_CV_min(self, system):
        CVs = []
        for i_CV in self.CVs_list:
            if i_CV[0] == 'inverse_r':
                CVs.append(self.inverse_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'exp_r':
                CVs.append(self.exp_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'inverse_r_exp_r':
                CVs.append(self.inverse_r_exp_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'distance':
                CVs.append(self.distance(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'central_R':
                CVs.append(self.central_R(system, i_CV[1]))
            elif i_CV[0] == 'min_distance':
                CVs.append(self.min_distance(system, i_CV[1]))
            elif i_CV[0] == 'max_distance':
                CVs.append(self.max_distance(system, i_CV[1]))
        CVs = np.array(CVs)
        min_CV = CVs[0]
        for i_CV in CVs[1:]:
            min_CV = np.minimum(min_CV, i_CV)
        return min_CV
    
    def generate_CV_max_list(self, system_list):
        if type(system_list) == type([]):
            return np.array([self.generate_CV_max(i_sys) for i_sys in system_list])
        else:
            return np.array([self.generate_CV_max(i_sys) for i_sys in [system_list]])
        
    def generate_CV_max(self, system):
        CVs = []
        for i_CV in self.CVs_list:
            if i_CV[0] == 'inverse_r':
                CVs.append(self.inverse_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'exp_r':
                CVs.append(self.exp_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'inverse_r_exp_r':
                CVs.append(self.inverse_r_exp_r(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'distance':
                CVs.append(self.distance(system, i_CV[1], i_CV[2]))
            elif i_CV[0] == 'central_R':
                CVs.append(self.central_R(system, i_CV[1]))
            elif i_CV[0] == 'min_distance':
                CVs.append(self.min_distance(system, i_CV[1]))
            elif i_CV[0] == 'max_distance':
                CVs.append(self.max_distance(system, i_CV[1]))
        CVs = np.array(CVs)
        max_CV = CVs[0]
        for i_CV in CVs[1:]:
            max_CV = np.maximum(max_CV, i_CV)
        return max_CV

    @staticmethod
    def central_R(system, atom_list):
        if type(atom_list) == int:
            atom_list = np.array([atom_list])
        elif type(atom_list) == list:
            atom_list = np.array(atom_list)
        elif type(atom_list) != np.ndarray:
            raise ValueError
        coordinates = system.get_positions()
        atom_list -= 1
        central_R_list = np.linalg.norm(coordinates[atom_list], axis=1)
        return central_R_list

    @staticmethod
    def distance(system, group_1, group_2):
        if type(group_1) == int:
            group_1 = np.array([group_1])
        elif type(group_1) == list:
            group_1 = np.array(group_1)
        elif type(group_1) != np.ndarray:
            raise ValueError
        if type(group_2) == int:
            group_2 = np.array([group_2])
        elif type(group_2) == list:
            group_2 = np.array(group_2)
        elif type(group_2) != np.ndarray:
            raise ValueError
        coordinates = system.get_positions()
        average_1_coordinate = np.sum(coordinates[group_1-1],axis=0)/len(group_1)
        average_2_coordinate = np.sum(coordinates[group_2-1],axis=0)/len(group_2)
        R_1_2 = np.linalg.norm(average_1_coordinate - average_2_coordinate)
        return R_1_2

    def inverse_r(self, system, group_1, group_2):
        R = self.distance(system, group_1, group_2)
        return 1/R

    def exp_r(self, system, group_1, group_2):
        R = self.distance(system, group_1, group_2)
        return np.exp(1/R)

    def inverse_r_exp_r(self, system, group_1, group_2):
        inverse_R = 1/self.distance(system, group_1, group_2)
        return np.array([inverse_R, np.exp(inverse_R)])

    def min_distance(self, system, group_list):
        R_list = []
        for group_i in group_list[:-1]:
            for group_j in group_list[1:]:
                R_i_j = self.distance(system, group_i, group_j)
                R_list.append(R_i_j)
        return np.min(np.array(R_list))

    def max_distance(self, system, group_list):
        R_list = []
        for group_i in group_list[:-1]:
            for group_j in group_list[1:]:
                R_i_j = self.distance(system, group_i, group_j)
                R_list.append(R_i_j)
        return np.max(np.array(R_list))
