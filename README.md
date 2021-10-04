# MoREST: a molecular reaction simulation toolkit
This is a package for chemical reaction simulation by using the statistical mechanics method. With API, this package can be used in combination with classical or *ab initio* molecular dynamics packages to simulate and analyze chemical reactions. The prepared potential energy surfaces can also be used with this package to study the dynamics of the system.

## *ab* *initio* statistical mechanics for reaction simulation
With enhanced sampling methods, the whole configurational space of the system will be traveled. The reaction networks and the reaction models will be constructed.


## Full dimensional *ab* *initio* reaction dynamics
Calculate the initial velocity of an incident molecule with particular vibration modes. Calculate the initial velocity of an incident molecule with rotation model, collision energy, and angle. 

## Units in MoREST
time: ps, length: $\overset{\circ}{\mathbb{A}}$, energy: eV, force: eV/$\overset{\circ}{\mathbb{A}}$, velocity: $\overset{\circ}{\mathbb{A}}$/ps, acceleration: $\overset{\circ}{\mathbb{A}}$/ps^2, mass: eV/c^2 = eV * $\overset{\circ}{\mathbb{A}}$^-2 * ps^2 (1 atomic mass (Dalton) = 1.1531805312624011e-21 eV $\overset{\circ}{\mathbb{A}}$^-2 ps^2)