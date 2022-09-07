from MoREST import morest

molpro_para_dict={}


md_job = morest(calculator=molpro_para_dict)
md_job.phase_space_sampling()
