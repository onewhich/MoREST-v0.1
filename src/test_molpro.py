from MoREST import morest
from ase.calculators.gaussian import Gaussian
'''
memory,12,g

symmetry,nosym
bohr
geometry={
 Al         -1.576726    0.000000    0.000000
 F          1.576726    -0.000000    0.000000
 Al         -0.000000    8.576726    0.000000
 F          0.000000    5.423274    -0.000000
}

basis=avqz

hf
ccsd(t)

'''

method=['hf','ccsd(t)']
basis='avqz'
calculator = Gaussian(label='molpro', cpu=cpu, mem='12,g', \
                      method=method, basis=basis, unit='Angstrom',nosymm='nosymm')


md_job = morest(calculator='molpro_sp')
md_job.phase_space_sampling()
