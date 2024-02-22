#! /usr/bin/env python

from ase.io import read, write
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase import units
import numpy as np
import sys

incident_init_xyz_list_filename = sys.argv[1]
incident_init_xyz_list = np.loadtxt(incident_init_xyz_list_filename)

incident_final_xyz_list_filename = sys.argv[2]
incident_final_xyz_list = np.loadtxt(incident_final_xyz_list_filename)

temperature = 10000 # velocity distribute at this temperature

atoms = read('incident.xyz',format='xyz')

incident_init_file=open("incident_init_file_"+str(temperature)+"K_"+str(len(incident_init_xyz_list)),'w')

for i_xyz in range(len(incident_init_xyz_list)):
	MaxwellBoltzmannDistribution(atoms, temperature * units.kB)
	#Stationary(atoms)  # zero linear momentum
	#ZeroRotation(atoms)  # zero angular momentum

	masses=atoms.get_masses()
	momenta=atoms.get_momenta()
	i_v=momenta/masses

#	v_rate = np.linalg.norm(i_v)
#	print(v_rate)
#	v_rate = np.linalg.norm(i_v) / ( units.m / units.s ) # output unit: m/s
#	print(str(v_rate)+" m/s")

	v_rate = np.linalg.norm(i_v) / ( units.Bohr / units.s ) # output unit: bohr/s
#	print(i_v / (units.Ang * units.Bohr) / (units.fs * 1e-12))
#	print(str(v_rate)+" Bohr/s")
#	print(-v_rate)


	incident_init_xyz = incident_init_xyz_list[i_xyz]
	incident_final_xyz = incident_final_xyz_list[i_xyz]

	incident_velocity = (incident_final_xyz-incident_init_xyz)/np.linalg.norm(incident_final_xyz-incident_init_xyz)*v_rate

	for i in range(3):
		incident_init_file.write(str(incident_init_xyz[i])+" ")
	for i in range(3):
		incident_init_file.write(str(incident_velocity[i])+" ")
	incident_init_file.write(str(v_rate)+"\n")
