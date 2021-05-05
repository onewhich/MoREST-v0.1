#! /usr/bin/env python

import numpy as np
import sys

# Calculate the relative velocity in CO vibrational mode under harmonic oscilator approximation

def velocity_of_vibrational_mode(nu):
	return 38.801929*np.sqrt(nu+0.5) # unit: AA/ps

if __name__ == '__main__':
	nu=int(sys.argv[1])
	print(velocity_of_vibrational_mode(nu))
