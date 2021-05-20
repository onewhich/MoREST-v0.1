export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
python builder.py
gfortran -o test test.f90 -L./ -lmorest
