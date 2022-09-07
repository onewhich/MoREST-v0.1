from runpy import run_module
from shutil import ExecError
from unittest import expectedFailure
import numpy as np
#import joblib
import pickle
from copy import deepcopy
from ase import units
import subprocess

class ml_potential:
    '''
    This class implements loading machine learned many body potential and returning the potential as the output of the Cartesian coordinates input.
    INPUT:
    trained_ml_potential: trained model in scikit-learn format and loaded by joblib.
    #model_features: (numpy npy file) dictionary including the name of the features used in machine learning, loaded by numpy.
    #model_labels: (numpy npy file) dictionary including the name of the labels used in machine learning, loaded by numpy.
    system: ase.Atoms object
    
    '''
    def __init__(self, trained_ml_potential, if_active_learning):
        #self.ml_potential = joblib.load(trained_ml_potential)
        self.ml_potential = pickle.load(open(trained_ml_potential, 'rb'))
        self.if_active_learning = if_active_learning
        #self.ml_features = np.load(model_features, allow_pickle=True)
        #self.ml_labels = np.load(model_labels, allow_pickle=True)
        
    @staticmethod
    def generate_Al2F2_representation(Al2F2):
    
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
                                  

        #E_Al = -241.93373718
        #E_F = -99.65284502
        #gpr_Al2 = joblib.load('GPR_Al2_inv_r_exp_r_Ebinding.joblib')
        #gpr_F2 = joblib.load('GPR_F2_inv_r_exp_r_Ebinding.joblib')
        #gpr_AlF = joblib.load('GPR_AlF_inv_r_exp_r_Ebinding.joblib')
        #gpr_AlF2 = joblib.load('GPR_AlF2_inv_r_exp_r_Ebinding.joblib')
        #gpr_Al2F = joblib.load('GPR_Al2F_inv_r_exp_r_Ebinding.joblib')

        #Eb_rAl2 = gpr_Al2.predict(np.array([[inverse_r_Al1_Al3,exp_r_Al1_Al3]],dtype=object))
        #Eb_rF2 = gpr_F2.predict(np.array([[inverse_r_F2_F4,exp_r_F2_F4]],dtype=object))
        #Eb_rAlF_1 = gpr_AlF.predict(np.array([[inverse_r_Al1_F2,exp_r_Al1_F2]],dtype=object))
        #Eb_rAlF_2 = gpr_AlF.predict(np.array([[inverse_r_Al3_F4,exp_r_Al3_F4]],dtype=object))
        #Eb_rAlF_3 = gpr_AlF.predict(np.array([[inverse_r_Al1_F4,exp_r_Al1_F4]],dtype=object))
        #Eb_rAlF_4 = gpr_AlF.predict(np.array([[inverse_r_Al3_F2,exp_r_Al3_F2]],dtype=object))
        
        #subfeature_AlF2_1 = np.array([[Eb_rAlF_1, inverse_r_Al1_F2, exp_r_Al1_F2, Eb_rAlF_3, inverse_r_Al1_F4, exp_r_Al1_F4, Eb_rF2, inverse_r_F2_F4, exp_r_F2_F4]],dtype=object)
        #subfeature_AlF2_2 = np.array([[Eb_rAlF_2, inverse_r_Al3_F4, exp_r_Al3_F4, Eb_rAlF_4, inverse_r_Al3_F2, exp_r_Al3_F2, Eb_rF2, inverse_r_F2_F4, exp_r_F2_F4]],dtype=object)
        #subfeature_Al2F_1 = np.array([[Eb_rAlF_1, inverse_r_Al1_F2, exp_r_Al1_F2, Eb_rAlF_4, inverse_r_Al3_F2, exp_r_Al3_F2, Eb_rAl2, inverse_r_Al1_Al3, exp_r_Al1_Al3]],dtype=object)
        #subfeature_Al2F_2 = np.array([[Eb_rAlF_2, inverse_r_Al3_F4, exp_r_Al3_F4, Eb_rAlF_3, inverse_r_Al1_F4, exp_r_Al1_F4, Eb_rAl2, inverse_r_Al1_Al3, exp_r_Al1_Al3]],dtype=object)
        
        #Eb_AlF2_1 = gpr_AlF2.predict(subfeature_AlF2_1)
        #Eb_AlF2_2 = gpr_AlF2.predict(subfeature_AlF2_2)
        #Eb_Al2F_1 = gpr_AlF2.predict(subfeature_Al2F_1)
        #Eb_Al2F_2 = gpr_AlF2.predict(subfeature_Al2F_2)
        
        #representation = np.array([E_Al, E_F,
        #                  Eb_rAl2, inverse_r_Al1_Al3, exp_r_Al1_Al3, Eb_rF2, inverse_r_F2_F4, exp_r_F2_F4,
        #                  Eb_rAlF_1, inverse_r_Al1_F2, exp_r_Al1_F2, Eb_rAlF_2, inverse_r_Al3_F4, exp_r_Al3_F4,
        #                  Eb_rAlF_3, inverse_r_Al1_F4, exp_r_Al1_F4, Eb_rAlF_4, inverse_r_Al3_F2, exp_r_Al3_F2,
        #                  Eb_AlF2_1, Eb_AlF2_2, Eb_Al2F_1, Eb_Al2F_2],dtype=object)
    
        return np.array(representation)

    def get_ml_potential(self, system_list):
        if type(system_list) != list:
            raise ValueError
        representation_list = [self.generate_Al2F2_representation(i_system) for i_system in system_list]
        ml_energy, ml_energy_std = self.ml_potential.predict(representation_list, return_std=True)
        ml_energy = np.array(ml_energy) * units.Hartree # change energy in Hartree to eV
        ml_energy_std = np.array(ml_energy_std) * units.Hartree
        return ml_energy, ml_energy_std

    def get_potential_FD_forces(self, system, displacement=0.0025, energy_difference_tolerance=0.01):
        system_list = [system]
        n_atoms = system.get_global_number_of_atoms()
        forces = []
        for i in range(n_atoms):
            for j in range(3):
                new_system = deepcopy(system)
                coordinates = new_system.get_positions()
                coordinates[i,j] = coordinates[i,j] + displacement
                new_system.set_positions(coordinates)
                system_list.append(new_system)
        # Get the predictions of energy and uncertainty
        energy_list, energy_std_list = self.get_ml_potential(system_list)
        #print("Energy:", energy_list)
        #print("Energy std:", energy_std_list)
        energy_0 = energy_list[0]
        energy_std_0 = energy_std_list[0]

        # Determine if the energy need to be calculated on the fly
        if self.if_active_learning and (energy_std_0 > energy_difference_tolerance):
            print("ML energy uncertainty is larger than tolerance(=", energy_difference_tolerance,"): ", energy_std_0)
            return float('nan'), float('nan')

        for i,i_energy in enumerate(energy_list[1:]):
            force_value = -1*(i_energy - energy_0)/displacement
            forces.append(force_value)
        forces = np.array(forces)
        #print(forces)
        return energy_0, forces.reshape(n_atoms, 3)


         
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


class molpro:
    def __init__(self, molpro_para_dict):
        self.molpro_dir = molpro_para_dict['molpro_dir']
        self.method = molpro_para_dict['method']
        self.basis = molpro_para_dict['basis']
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
        self.elements = system.get_chemical_symbols()
        self.positions = system.get_positions()
        self.run_molpro()

        return self.potential_energy, self.forces

    def run_molpro(self):
        runcommand = self.molpro_dir + " < " + self.infile + " > " + self.outfile

        inpstr = 'memory,'+self.memory+'\n\n'
        inpstr += 'symmetry,nosym\n'

        inpstr += self.unit + '\n'

        # Parse the geometry
        inpstr += 'geometry={\n'
        for i,element in enumerate(self.elements):
            inpstr += element
            inpstr += ' '
            for j in range(3):
                inpstr += ' ' + str(self.positions[i][j])
            inpstr += '\n'

        inpstr += '}'

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
        #print("Molpro exit code:", runresult.returncode)
        return runresult.returncode
<<<<<<< HEAD
=======
def get_xyz_energy(file, if_get_force = False):
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

           - Force:

             CCSD(T) GRADIENT FOR STATE 1.1

             Atom          dE/dx               dE/dy               dE/dz

             1         0.000000000        -0.000000000        -0.070539245
             2        -0.000000000         0.000000000         0.070539245

             Nuclear force contribution to virial =         0.266599708




    """

    unit_xyz = "Bohr"
    unit_energy = "Hartree"
    
    print("\n\nFile:", file)
    path = ""
    
    with open(file,'r') as f:
        path = file
        lines = f.readlines()
        
        
    
        energy = float('nan')
        xs = []
        ys = []
        zs = []
        elements = []
        xyz = []
        force = []

 
        if(len(lines)<1):
            energy = float('Inf')
            return path, elements, xyz_Al1F2Al3F4, energy
    
            
        if(lines[-1].find("terminated")==-1):
            energy = float('Inf')
            return path, elements, xyz_Al1F2Al3F4, energy
    

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
 
        
        for i, line in enumerate(lines):
            


            if(line.find("ATOMIC COORDINATES")!=-1):
                ii = i+4
                xs = []
                ys = []
                zs = []
                elements = []
                while (lines[ii].find("Bond lengths")==-1) and (len(lines[ii].split())>3):
                    element = lines[ii].split()[1]
                    x = float(lines[ii].split()[-3])
                    y = float(lines[ii].split()[-2])
                    z = float(lines[ii].split()[-1])
                    elements.append(element)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    ii += 1
                
                
            """
            Energy:
            
            CCSD(T)/aug-cc-pVQZ energy=   -671.623485056226
            
            """
            if(line.find("Bond lengths in ")!= -1):
                print("Unit of coordinates read from the output file:", line.split("Bond lengths in ")[-1].split(" ")[0])
                
                
            if(line.find(" energy=")!=-1):
                energy = float(line.split("=")[-1])
        
        
        
        
        

        for i_atom in range(len(xs)):
        
            xyz.append([xs[i_atom],ys[i_atom],zs[i_atom]])

        
        print(" \nElements:", elements, "; Coordinates:", xyz, "; \n Energy:", energy, ";\n Force:", force)


        if if_get_force:
            return path, elements, xyz, energy, force
        else:
            return path, elements, xyz, energy

>>>>>>> 016ec7614b437f82d4021d02c82c3dd8ad79625b

