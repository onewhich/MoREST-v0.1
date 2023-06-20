# including active learning

from read_parameters import read_parameters
from many_body_potential import Molpro
from initialization import initialize_calculator
from structure import read_xyz_file
from ase.neb import NEB
from ase.optimize import MDMin

molpro_dir = "/home/wang/software/molpro/bin/molpro"
memory = "20,g"
ntasks = 1
nthreads = 64
method = "hf\nmp2\nforce\nccsd(t)"
basis = "avqz"
ab_initio_calculator = Molpro(molpro_dir=molpro_dir, ntasks=ntasks, nthreads=nthreads, method=method, basis=basis)

MoREST_parameters = read_parameters(parameter_file='MoREST.in')
morest_parameters = MoREST_parameters.get_morest_parameters()

ml_calculator = initialize_calculator(morest_parameters, ab_initio_calculator, 'MoREST.log').get_current_calculator()


# Read initial and final states:
initial = read_xyz_file('initial.xyz')
final = read_xyz_file('final.xyz')
# Make a band consisting of 5 images:
images = [initial]
images += [initial.copy() for i in range(3)]
images += [final]
neb = NEB(images)
# Interpolate linearly the potisions of the three middle images:
neb.interpolate()
# Set calculators:
for image in images[1:4]:
    image.calc = ml_calculator
# Optimize:
optimizer = MDMin(neb, trajectory='A2B.traj')
optimizer.run(fmax=0.04)