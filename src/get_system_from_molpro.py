#! /usr/bin/env python

# get_system_from_molpro.py molpro_out_file_list.txt filename.xyz

import sys
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.atoms import Atoms
from ase.io import write
from ase import units

def parse_outfile(file):
    lines = open(file, 'r').readlines()
    energy = np.nan
    atomic_number = []
    forces = []
    positions = []
    n_atom = 0
    if( lines[-1].find("terminated") !=-1 ):
        for i, line in enumerate(lines):
            if(line.find("NR  ATOM    CHARGE       X              Y              Z")!=-1):
                ii = i+2
                while len(lines[ii].split()) > 0:
                    string_line = lines[ii].split()
                    if n_atom < int(string_line[0]):
                        n_atom = int(string_line[0])
                        tmp_pos = []
                        atomic_number.append(int(float(string_line[2])))
                        tmp_pos.append(float(string_line[3]))
                        tmp_pos.append(float(string_line[4]))
                        tmp_pos.append(float(string_line[5]))
                        positions.append(np.array(tmp_pos))
                    ii += 1
            elif(line.find("GRADIENT FOR STATE")!=-1):
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
                forces = []
                for i_atom in range(len(forcexs)):
                    forces.append([forcexs[i_atom],forceys[i_atom],forcezs[i_atom]])
            elif(line.find(" energy=")!=-1):
                energy = float(line.split("=")[-1])
        if n_atom == 0:
            print("Error: No atoms found in file (" + file + ").")
            return [], np.array([]), np.nan, np.array([])
        else:
            return atomic_number, np.array(positions) * units.Bohr, energy * units.Hartree, np.array(forces) * (units.Hartree/units.Bohr) * -1
    else:
        print("Error: The file (" + file + ") does not seem to be a finished Molpro output file.")
        return [], np.array([]), np.nan, np.array([])

file_list = open(sys.argv[1],'r').readlines()
molpro_systems = []

for line in file_list:
    molpro_out = line.split()[0]

    atomic_number, positions, energy, forces = parse_outfile(molpro_out)
    system = Atoms(atomic_number, positions=positions)
    if len(atomic_number) > 0:
        if len(forces) > 0:
            system.calc = SinglePointCalculator(system, energy=energy)
        else:
            system.calc = SinglePointCalculator(system, energy=energy, forces=forces)
        molpro_systems.append(system)

write(sys.argv[2], molpro_systems, format='extxyz')
