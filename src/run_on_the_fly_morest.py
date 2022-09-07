from MoREST import morest
from ase.calculators.gaussian import Gaussian

cpu='0-39'
charge=1
mult=1
method='MN15'
basisfile='../../../def2-TZVPD'
calculator = Gaussian(label='gjf', cpu=cpu, mem='12GB', chk='gjf.chk', charge=charge, mult=mult,\
                      method=method, basisfile=basisfile, geom='NoCrowd', scf='qc', nosymm='nosymm')


md_job = morest(calculator=calculator)
md_job.phase_space_sampling()
