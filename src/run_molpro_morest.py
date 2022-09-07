from MoREST import morest

molpro_para_dict={}
molpro_para_dict['molpro_dir'] = "/home/wang/software/molpro/bin/molpro"
molpro_para_dict['method'] = "hf\nccsd(t)\nforce, numerical"
molpro_para_dict['basis'] = "avqz"

md_job = morest(calculator=molpro_para_dict)
md_job.phase_space_sampling()
