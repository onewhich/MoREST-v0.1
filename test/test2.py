from MoREST import morest
from ase.calculators.gaussian import Gaussian

calculator = Gaussian(label='gjf', cpu='0-39', mem='12GB', chk='gjf.chk', charge=0, \
                      method='b3lyp', basis='def2svp', geom='NoCrowd', scf='yqc')
Al2F2_10 = morest()
Al2F2_10.phase_space_sampling(calculator=calculator)