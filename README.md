# MoREST: Molecular REaction Simulation Toolkit
This is a package for chemical reaction simulation by using the statistical mechanics method. With API, this package can be used in combination with classical or *ab initio* molecular dynamics packages to simulate and analyze chemical reactions. The prepared potential energy surfaces can also be used with this package to study the dynamics of the system.

## *ab* *initio* statistical mechanics for reaction simulation
With enhanced sampling methods, the whole configurational space of the system will be traveled. The reaction networks and the reaction models will be constructed.


## Full dimensional *ab* *initio* reaction dynamics
Calculate the initial velocity of an incident molecule with particular vibration modes. Calculate the initial velocity of an incident molecule with rotation model, collision energy, and angle. 

## Machine learning potential parameters
MoREST supports multiple ML potential backends. The parameter `ML_model_type` controls the model:
* `gpr` (default): GaussianProcessRegressor with uncertainty.
* `rfr`: RandomForestRegressor with ensemble-based uncertainty.
* `bayesian_ridge`: BayesianRidge with predictive standard deviations.

## Units in MoREST (Discarded)
time: ps, length: AA, energy: eV, force: eV/AA, velocity: AA/ps, acceleration: AA/ps^2, mass: eV/c^2 = eV * AA^-2 * ps^2 (1 atomic unit mass (Dalton) = 1.1531805312624011e-21 eV AA^-2 ps^2)
