#! /usr/bin/env python

import sys
sys.path.append('../../Vibration-mode/src')
import CO_vib_HO as CO_vib
import numpy as np

# CO collide along z axis
# v_translate: translational velocity in AA/ps
# omega_rotate: rotational velocity in rad/ps
# nu: quantum number of vibrational state

def center_of_mass(CO_xyz):
	mass_C = 12.
	mass_O = 16.
	return (CO_xyz[0]*mass_C+CO_xyz[1]*mass_O)/(mass_C+mass_O)

def CO_initialize(v_translate=3.5,direction_translate=np.array([0.0,0.0,-1.0]),omega_rotate=0,direction_rotate=np.array([1.0,0.0,0.0]),nu=0,CO_xyz=np.array([[0.0,0.0,0.0],[0.0,0.0,1.13543716]])):

	mass_C = 12.
	mass_O = 16.

	direction_translate = direction_translate/np.linalg.norm(direction_translate) # make sure the direction vector is unit vector
	direction_rotate = direction_rotate/np.linalg.norm(direction_rotate)	  # make sure the direction vector is unit vector
	# print(direction_translate,direction_rotate)


	# calculate the collision velocity of the molecule
	v_molecule = v_translate * direction_translate

	# calculate the rotation velocity of each atom in the molecule
	mol_center = center_of_mass(CO_xyz)
	v_rotate = omega_rotate * direction_rotate
	r_atom = []
	v_atom = []
	for i_atom in CO_xyz:
		r_i_atom = i_atom - mol_center
		r_atom.append(r_i_atom)
		v_atom.append(np.cross(v_rotate,r_i_atom))
	r_atom = np.array(r_atom)
	v_atom = np.array(v_atom)


	# calculate the relative velocity of vibration in the molecule
	v_vibrate = CO_vib.velocity_of_vibrational_mode(nu) * (CO_xyz[1] - CO_xyz[0]) / np.linalg.norm(CO_xyz[1] - CO_xyz[0])
	
	# calculate total velocity of each atom in the molecule
	for i in range(len(v_atom)):
		v_atom[i] = v_atom[i] + v_molecule
	v_atom[0] = v_atom[0] - v_vibrate * mass_O/(mass_C+mass_O)
	v_atom[1] = v_atom[1] + v_vibrate * mass_C/(mass_C+mass_O)

	return v_atom

if __name__ == '__main__':

	CO_xyz=np.array([[0.0,0.0,0.0],[0.0,0.0,1.13543716]])
	direction_rotate=np.array([1.0,0.0,0.0])
	for omega_rotate in [0.0,1.0,2.0,3.0]:
		for nu in [0,5,10]:
			print(CO_initialize(v_translate=3.5,direction_translate=np.array([0.0,0.0,-1.0]),omega_rotate=omega_rotate,direction_rotate=direction_rotate,nu=nu,CO_xyz=CO_xyz))

	CO_xyz=np.array([[0.0,0.0,0.0],[1.13543716,0.0,0.0]])
	direction_rotate=np.array([0.0,0.0,1.0])
	for omega_rotate in [0.0,1.0,2.0,3.0]:
		for nu in [0,5,10]:
			print(CO_initialize(v_translate=3.5,direction_translate=np.array([0.0,0.0,-1.0]),omega_rotate=omega_rotate,direction_rotate=direction_rotate,nu=nu,CO_xyz=CO_xyz))
