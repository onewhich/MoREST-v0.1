#export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
python builder.py
gfortran -c API_MoREST.f90
gfortran -o test test_use_module.f90 API_MoREST.o -L../../../lib -lmorest
