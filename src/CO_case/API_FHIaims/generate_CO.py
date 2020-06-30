#! /usr/bin/env python

import sys,os
sys.path.append('../../Initialize-collision-state/src')
import CO_initialize_HO as CO_init
import numpy as np

# generate CO coordinate and initial velocity in FHI-aims geometry.in format

target_geometry = open('geometry.in.target','r').readlines()



CO_xyz=np.array([[2.35484,2.87023,7.0],[2.35484,2.87023,8.13543716]])
direction_rotate=np.array([1.0,0.0,0.0])
for omega_rotate in [0.0,5.0,10.0]:
	for nu in [0,5,10]:
		initial_v = CO_init.CO_initialize(v_translate=3.5,direction_translate=np.array([0.0,0.0,-1.0]),omega_rotate=omega_rotate,direction_rotate=direction_rotate,nu=nu,CO_xyz=CO_xyz)
		foldername = 'around_x_rotate_'+str(omega_rotate)+'_vibrate_'+str(nu)
		os.system('mkdir '+foldername)
		geometry_path = os.path.join(foldername,'geometry.in')
		geometry_in = open(geometry_path,'w')
		for line in target_geometry:
			geometry_in.write(line)
		geometry_in.write('atom	'+str(CO_xyz[0,0])+'      '+str(CO_xyz[0,1])+'      '+str(CO_xyz[0,2])+' C\n')
		geometry_in.write('  velocity      '+str(initial_v[0,0])+'      '+str(initial_v[0,1])+'      '+str(initial_v[0,2])+'\n')
		geometry_in.write('atom	'+str(CO_xyz[1,0])+'      '+str(CO_xyz[1,1])+'      '+str(CO_xyz[1,2])+' O\n')
		geometry_in.write('  velocity      '+str(initial_v[1,0])+'      '+str(initial_v[1,1])+'      '+str(initial_v[1,2])+'\n')
		geometry_in.close()
			

CO_xyz=np.array([[2.35484,2.87023,7.0],[3.49027716,2.87023,7.0]])
direction_rotate=np.array([0.0,0.0,1.0])
for omega_rotate in [0.0,5.0,10.0]:
	for nu in [0,5,10]:
		initial_v = CO_init.CO_initialize(v_translate=3.5,direction_translate=np.array([0.0,0.0,-1.0]),omega_rotate=omega_rotate,direction_rotate=direction_rotate,nu=nu,CO_xyz=CO_xyz)
		foldername = 'around_z_rotate_'+str(omega_rotate)+'_vibrate_'+str(nu)
		os.system('mkdir '+foldername)
		geometry_path = os.path.join(foldername,'geometry.in')
		geometry_in = open(geometry_path,'w')
		for line in target_geometry:
			geometry_in.write(line)
		geometry_in.write('atom	'+str(CO_xyz[0,0])+'      '+str(CO_xyz[0,1])+'      '+str(CO_xyz[0,2])+' C\n')
		geometry_in.write('  velocity      '+str(initial_v[0,0])+'      '+str(initial_v[0,1])+'      '+str(initial_v[0,2])+'\n')
		geometry_in.write('atom	'+str(CO_xyz[1,0])+'      '+str(CO_xyz[1,1])+'      '+str(CO_xyz[1,2])+' O\n')
		geometry_in.write('  velocity      '+str(initial_v[1,0])+'      '+str(initial_v[1,1])+'      '+str(initial_v[1,2])+'\n')
		geometry_in.close()
