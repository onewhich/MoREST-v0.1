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
    