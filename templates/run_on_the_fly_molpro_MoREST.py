from MoREST.main import morest
from MoREST.many_body_potential import Molpro

molpro_dir = "/home/wang/software/molpro/bin/molpro"
ntasks = 1
nthreads = 32
method = "hf\nccsd(t)\nforce"
basis = "avqz"
calculator = Molpro(molpro_dir=molpro_dir, ntasks=ntasks, nthreads=nthreads, method=method, basis=basis)

md_job = morest(calculator=calculator)
md_job.phase_space_sampling()
