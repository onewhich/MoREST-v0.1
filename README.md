# MoREST v0.1

**MoREST** (**Mo**lecular **RE**action **S**imulation **T**oolkit) is a Python toolkit for atomistic reaction simulation and analysis.  
From the current codebase, this release is organized as a Python API around the `morest` class in `src/MoREST/main.py`, with a central text configuration file (`MoREST.in`) that switches between simulation modes such as sampling, scattering, rovibrational dynamics, structure optimization, enhanced sampling, machine-learning potentials, and wall/barostat constraints.

This README is a complete replacement for the original repository README and is written from the observable repository structure, installation metadata, and the shipped `templates/MoREST.in`.

---

## What MoREST can do

MoREST is built around several major workflows.

### 1. Phase-space sampling
MoREST can initialize and run phase-space exploration for molecular systems using:
- **MD**
- **RPMD**
- **RPMD_NM**
- **MC** (listed in the input template as a supported sampling method)

For MD-style sampling, the template exposes multiple ensembles and thermostats/barostats, including:
- `NVE_VV`
- `NVK_VR`
- `NVT_Berendsen`
- `NVT_Langevin`
- `NVT_SVR`
- `NPH_SVR`
- `NPT_Berendsen`
- `NPT_Langevin`
- `NPT_SVR`

### 2. Trajectory scattering
MoREST supports trajectory-based scattering simulations for:
- **molecule–molecule**
- **molecule–surface/slab**

The template allows control over:
- number of trajectories
- collision velocity or collision energy
- target and incident temperatures
- geometry initialization on spheres or above surfaces
- fixed atoms
- stop conditions based on collective variables such as distance, angle, and distance-to-center

### 3. Rovibrational dynamics
A dedicated rovibrational mode is present for generating molecule-only dynamics without an incident collision partner.  
The template indicates this mode is intended for **rovibrational spectrum-related calculations**.

### 4. Structure searching / optimization
MoREST includes a structure-searching workflow with local, transition-state, and global-search-style optimizers exposed in the parameter file, including:
- `GD`
- `CG`
- `BFGS`
- `L-BFGS`
- `L-BFGS-B`
- `FIRE`
- `RP_BFGS`
- `BFGS-TS`
- `L-BFGS-TS`
- `dimer`
- `GAD`
- `SGD`
- `Adam`

### 5. Enhanced sampling
The main driver and the template show support for enhanced sampling, especially:
- **RE** (Replica Exchange)
- **ITS** (Integrated Tempering Sampling)

The input template also lists additional methods as planned/available interface options, such as:
- `pRE`
- `ST`
- `aMD`
- `Wang-Landau`
- `NS`
- `TPS`
- `TIS`
- `BH`
- `MH`
- `SA`

### 6. Machine-learning potentials
MoREST can use machine-learning potentials through the `ML_potential` backend and supports at least these model families:
- **GPR** (`gpr`)
- **Random Forest Regressor** (`rfr`)
- **Bayesian Ridge** (`bayesian_ridge`)

The template additionally shows support for several feature/representation choices, including:
- Cartesian coordinates
- distance/inverse-distance based descriptors
- adjacency-matrix style encodings
- `PIP_global`
- optional `MBTR`
- optional `SOAP`
- optional `MACE`

### 7. Active learning and uncertainty-aware runs
For ML-backed runs, the parameter file exposes:
- uncertainty printing
- finite-difference force evaluation
- active-learning switches
- energy-uncertainty tolerances
- retraining/appending intervals for newly sampled configurations

### 8. External calculators and on-the-fly ab initio evaluation
The constructor docstring in `main.py` states that a calculator is required when:
- `Many_body_potential` is set to **`on_the_fly`**
- or active learning requires fresh reference calculations

This suggests MoREST is meant to interoperate with **ASE-compatible calculators** and external electronic-structure workflows.

---

## Repository layout

A minimal observable repository layout is:

```text
MoREST-v0.1/
├── src/
│   └── MoREST/
├── templates/
│   └── MoREST.in
├── utilities/
├── requirements.txt
├── setup.py
└── README.md
```

### Important files
- `src/MoREST/main.py`  
  Main Python entry point containing the `morest` class and the top-level workflows.
- `src/MoREST/read_parameters.py`  
  Reads and stores parameters from `MoREST.in`.
- `templates/MoREST.in`  
  Full example/template input file for configuring a run.
- `requirements.txt` / `setup.py`  
  Installation metadata and core dependencies.

---

## Installation

## Requirements

The repository declares the following core Python dependencies:

```text
numpy
scipy
ase
tqdm
scikit-learn
MoREAT
```

Install them with:

```bash
pip install -r requirements.txt
```

Then install MoREST itself:

```bash
pip install -e .
```

### Notes
- `MoREAT` is listed as a required dependency by both `requirements.txt` and `setup.py`. Make sure it is available in your environment before importing MoREST.
- Some representations in `MoREST.in` mention extra ecosystems such as **DScribe** (`MBTR`, `SOAP`) and **MACE**. Those should be treated as **optional extras required only if you choose those representations/models**.
- For `Many_body_potential on_the_fly`, you will typically also need an **ASE calculator** connected to your electronic-structure code.

---

## How the Python API is structured

The central object is the `morest` class in `MoREST.main`.

### Constructor

```python
from MoREST.main import morest

job = morest(parameter_file="MoREST.in", calculator=None)
```

### Main execution methods

Depending on the switches inside `MoREST.in`, the object initializes one or more workflows and exposes methods such as:

```python
job.phase_space_sampling()
job.trajectory_scattering()
job.molecule_rovibrating()
job.structure_searching()
```

The source also shows internal support for:
- replica exchange (`enhanced_sampling_RE`)
- integrated tempering (`enhanced_sampling_ITS`)
- wall-potential evaluation

---

## Quick start

## 1. Prepare the input file

Copy the shipped template and edit it:

```bash
cp templates/MoREST.in ./MoREST.in
```

At minimum, decide:
- which workflow to turn on:
  - `Phase_space_sampling`
  - `Trajectory_scattering`
  - `Molecule_rovibrating`
  - `Structure_searching`
- which potential backend to use:
  - `ML_potential`
  - `on_the_fly`
  - `molpro`
- which structure files are needed:
  - `MoREST_sampling.xyz`
  - `MoREST_scattering_target.xyz`
  - `MoREST_scattering_incident.xyz`
  - `MoREST_rovibration.xyz`
  - `MoREST_searching.xyz`
  - `training_set.xyz`
  - external QC input such as `gjf.com`

## 2. Write a small driver script

### Example A: ML-backed phase-space sampling

```python
from MoREST.main import morest

job = morest(parameter_file="MoREST.in")
job.phase_space_sampling()
```

### Example B: on-the-fly electronic-structure calculations with an ASE calculator

```python
from ase.calculators.emt import EMT
from MoREST.main import morest

calc = EMT()  # replace with your actual ASE-compatible calculator
job = morest(parameter_file="MoREST.in", calculator=calc)
job.phase_space_sampling()
```

### Example C: trajectory scattering

```python
from MoREST.main import morest

job = morest(parameter_file="MoREST.in")
job.trajectory_scattering()
```

### Example D: structure search / optimization

```python
from MoREST.main import morest

job = morest(parameter_file="MoREST.in")
job.structure_searching()
```

---

## Understanding `MoREST.in`

The shipped `templates/MoREST.in` is the real control center of this package.  
Below is a practical guide to the major blocks.

## 1. Global control flags

Examples:
- `MoREST_initialization`
- `MoREST_save_parameters_file`
- `MoREST_load_parameters_file`

These control whether a run starts fresh or continues/appends, and whether parameter snapshots are written/read.

---

## 2. Potential backend

### `Many_body_potential`
Controls how energies and forces are obtained.

Observed values in the template:
- `ML_potential`
- `on_the_fly`
- `molpro`

### `Input_file`
External quantum-chemistry input file, for example:

```text
gjf.com
```

### `ML_potential_model`
Path to a trained ML model on disk.

### `ML_training_set`
Trajectory/training-set file, typically in XYZ/extxyz style.

---

## 3. Machine-learning options

### Model family
Use:

```text
ML_model_type GPR
```

Observed options:
- `GPR`
- `RFR`
- `bayesian_ridge`

### Feature representation
Use:

```text
ML_representation inverse_r_exp_r
```

Representations documented inside the shipped template include:
- `Cartesian`
- `inverse_r_exp_r`
- `inverse_r_exp_r_unsorted`
- `inverse_r`
- `inverse_r_unsorted`
- `distance_matrix`
- `adjacency_matrix`
- `adjacency_matrix_full`
- `adjacency_matrix_symmetric_expansion`
- `PIP_global`
- `MBTR`
- `SOAP`
- `MACE`
- `generate_Al2F2_representation`

### Extra ML features
The template also supports:
- additional user-defined features
- min-selected feature groups
- max-selected feature groups

### Uncertainty / active learning
Important controls include:
- `ML_print_uncertainty`
- `ML_FD_forces`
- `FD_displacement`
- `ML_active_learning`
- `ML_energy_uncertainty_tolerance`
- `ML_appending_set_number`
- `ML_appending_sampling_steps`
- `ML_GPR_noise_level_bounds`

---

## 4. Phase-space sampling block

Turn on:
```text
Phase_space_sampling True
```

Important parameters include:
- `Sampling_molecule`
- `Sampling_method`
- `Sampling_initial_T`
- `Sampling_initial_E`
- `Sampling_ensemble`
- `Sampling_traj_interval`

For MD:
- `MD_time_step`
- `MD_simulation_time`
- `MD_temperature`
- atom-fixing options:
  - `MD_fix_atoms_all`
  - `MD_fix_atoms_x`
  - `MD_fix_atoms_y`
  - `MD_fix_atoms_z`
- cleanup options:
  - `MD_clean_translation`
  - `MD_clean_rotation`

For RPMD:
- `RPMD_number_of_beads`
- `RPMD_beads_file`
- `RPMD_time_step`
- `RPMD_simulation_time`
- `RPMD_temperature`

---

## 5. Barostat / confined-space controls

The template contains a dedicated block for pressure control through bounded spaces:
- `Barostat_number`
- `Barostat_pressure`
- `Barostat_space_shape`
- `Barostat_space_type`
- `Barostat_space_size`
- `Barostat_action_atoms`

Observed shapes:
- `sphere`
- `cuboid`
- `plane`

This is especially relevant for confined systems and surface-like geometries.

---

## 6. Trajectory scattering block

Turn on:
```text
Trajectory_scattering True
```

Key parameters:
- `Scattering_type`
- `Scattering_traj_number`
- `Scattering_method`
- `Scattering_time_step`
- `Scattering_V_collision`
- `Scattering_E_collision`
- `Scattering_T_target`
- `Scattering_T_incident`
- `Scattering_target_molecule`
- `Scattering_incident_molecule`
- `Scattering_R_target`
- `Scattering_R_incident`
- `Scattering_fix_target`
- `Scattering_fix_incident`
- `Scattering_clean_rotation`

The stop-condition system is particularly important:
- `Scattering_stops_number`
- `Scattering_traj_stop`

The template documents stop conditions such as:
- `central_R_one`
- `central_R_all`
- `distance`
- `angle`
- `None`

This means a trajectory can be terminated automatically when a collective-variable criterion is met.

---

## 7. Rovibrational dynamics block

Turn on:
```text
Molecule_rovibrating True
```

Key parameters:
- `Rovibrating_method`
- `Rovibrating_molecule`
- `Rovibrating_time_step`
- `Rovibrating_traj_interval`
- `Rovibrating_simulation_time`
- `Rovibrating_vibration_T`
- `Rovibrating_vibration_E`
- `Rovibrating_rotation_T`
- `Rovibrating_rotation_E`
- `Rovibrating_clean_rotation`

This mode is described in the template as a scattering-like setup without an incident molecule, intended for rovibrational studies.

---

## 8. Structure search / optimization block

Turn on:
```text
Structure_searching True
```

Key parameters:
- `Searching_starting_point`
- `Searching_convergence`
- `Searching_max_steps`
- `Searching_constrained`
- `Searching_method`

Additional method-specific settings visible in the template:
- `Gradient_step_size`
- `LBFGS_history_step`
- `FIRE_equal_masses`
- `FIRE_time_step`
- `FIRE_max_time_step`
- `FIRE_alpha_init`
- `FIRE_N_min`
- `FIRE_f_increase`
- `FIRE_f_decrease`
- `FIRE_f_alpha`
- `RP_number_of_beads`
- `RP_temperature`
- `Dimer_distance`
- `Dimer_rotation_step`
- `GAD_time_step`
- `SGD_fraction`
- `Adam_beta1`
- `Adam_beta2`
- `Adam_eps`

---

## 9. Enhanced sampling block

Turn on:
```text
Enhanced_sampling True
```

Observed primary methods:
- `RE`
- `ITS`

### Replica Exchange (RE)
Parameters include:
- `RE_initialization`
- `RE_lower_bound_temperature`
- `RE_upper_bound_temperature`
- `RE_number_of_replica`
- `RE_replica_arrange`
- `RE_replica_temperatures`
- `RE_swap_interval`
- `RE_init_structures_list`
- `RE_energy_shift`

### Integrated Tempering Sampling (ITS)
Parameters include:
- `ITS_initialization`
- `ITS_lower_bound_temperature`
- `ITS_upper_bound_temperature`
- `ITS_number_of_replica`
- `ITS_replica_arrange`
- `ITS_replica_temperatures`
- `ITS_initial_nk`
- `ITS_pk0`
- `ITS_trial_MD_steps`
- `ITS_delta_pk`
- `ITS_weight_pk`
- `ITS_energy_shift`

---

## 10. Wall potential block

Turn on:
```text
Wall_potential True
```

Parameters include:
- `Wall_number`
- `Wall_collective_variable`
- `Wall_shape`
- `Wall_type`
- `Wall_scaling`
- `Wall_scope`
- `Wall_action_atoms`

Observed wall shapes:
- `spherical`
- `planar`
- `dot`
- `linear`
- `circular`

Observed wall types:
- `opaque_wall`
- `translucent_wall`
- `power_wall`

Geometry parameters include, for example:
- `Spherical_wall_center`
- `Spherical_wall_radius`
- `Planar_wall_point`
- `Planar_wall_normal_vector`
- `Dot_wall_position`

The template also shows how to define **multiple walls** by repeating the relevant parameters.

---

## Minimal example input files

## Example 1: short ML-driven MD sampling

```text
MoREST_initialization True
Many_body_potential ML_potential

ML_potential_model ./model.joblib
ML_training_set training_set.xyz
ML_model_type GPR
ML_representation inverse_r_exp_r
ML_active_learning False

Phase_space_sampling True
Sampling_initialization False
Sampling_molecule MoREST_sampling.xyz
Sampling_method MD
Sampling_ensemble NVT_SVR 10
Sampling_traj_interval 10

MD_time_step 1
MD_simulation_time 5000
MD_temperature 300
MD_clean_translation True
MD_clean_rotation True

Trajectory_scattering False
Molecule_rovibrating False
Structure_searching False
Enhanced_sampling False
Wall_potential False
```

## Example 2: structure optimization with an external calculator

```text
MoREST_initialization True
Many_body_potential on_the_fly
Input_file gjf.com

Phase_space_sampling False
Trajectory_scattering False
Molecule_rovibrating False

Structure_searching True
Searching_initialization False
Searching_starting_point MoREST_searching.xyz
Searching_method FIRE
Searching_convergence 5e-3
Searching_max_steps 200

Enhanced_sampling False
Wall_potential False
```

---

## Common output and runtime files

From the source and template, common files in a run may include:

- `MoREST.log`  
  Main log file opened by the constructor.
- `MoREST_sampling.xyz`
- `MoREST_scattering_target.xyz`
- `MoREST_scattering_incident.xyz`
- `MoREST_rovibration.xyz`
- `MoREST_searching.xyz`
- `MoREST_RPMD_beads.xyz`
- `training_set.xyz`

The repository `.gitignore` also suggests that generated or restart-like files may appear during workflows, for example:
- `*.log`
- `*.traj`
- `*.joblib`
- `*.str_new`

---

## Practical workflow suggestions

### If you want to run a pure ML trajectory
Use:
- `Many_body_potential ML_potential`
- a valid `ML_potential_model`
- an appropriate `ML_representation`
- `ML_FD_forces` as needed if analytic forces are not provided by your model

### If you want uncertainty-triggered refinement
Use:
- `ML_active_learning True`
- an uncertainty-aware model such as `GPR`
- `ML_energy_uncertainty_tolerance`
- appending/retraining controls

### If you want direct ab initio evaluation
Use:
- `Many_body_potential on_the_fly`
- pass an ASE-compatible calculator into the Python constructor

### If you want scattering statistics
Use:
- `Trajectory_scattering True`
- define target and incident structures
- choose `Scattering_type`
- define stop conditions carefully

### If you want structural relaxation or transition-state exploration
Use:
- `Structure_searching True`
- choose `Searching_method`
- start from a meaningful geometry
- set convergence and maximum-step limits conservatively

---

## Current caveats

- The distributed metadata does **not** define an obvious command-line entry point. The safest assumption is that this release is intended to be run through the **Python API**.
- Several template comments explicitly contain `TODO` markers. Some options may be experimental, partially implemented, or intended for future extension.
- Optional ML representations such as `MBTR`, `SOAP`, and `MACE` likely require external packages beyond the six core dependencies listed in `requirements.txt`.
- For `on_the_fly` calculations and some active-learning workflows, the quality and behavior of your run will depend heavily on the external ASE calculator and upstream electronic-structure settings.
- The old README contains a discarded unit statement. For practical use, follow the units documented inline in `MoREST.in` parameter comments, where values are mostly given in **fs**, **K**, **eV**, **Å**, and **bar**, depending on the parameter.

---

## License

This repository is released under the **GPL-3.0** license.

---

## Recommended first steps for a new user

1. Install dependencies and `pip install -e .`
2. Copy `templates/MoREST.in` into your working directory
3. Turn on **only one primary workflow** at a time:
   - sampling
   - scattering
   - rovibration
   - structure searching
4. Start with the simplest possible test system and a short run
5. Verify that:
   - the required XYZ/input files exist
   - the calculator or ML model path is valid
   - `MoREST.log` is being written
6. Expand to enhanced sampling, active learning, or wall/barostat constraints only after the base workflow runs cleanly

---

## Suggested citation / project description text

If you need one short project description for proposals, websites, or repository indexing, you can use:

> **MoREST** is a molecular reaction simulation toolkit for atomistic sampling, trajectory scattering, rovibrational dynamics, structure searching, enhanced sampling, and machine-learning-assisted potential evaluation, with a text-driven workflow configured through `MoREST.in`.

