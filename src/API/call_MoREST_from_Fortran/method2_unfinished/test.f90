program test_MoREST
    use forpy_mod
    implicit none

    integer :: ierror, i
    real(8) :: simulation_temperature = 798 ! K
    integer :: current_md_step = 0
    real(8) :: potential_energy
    integer :: if_initial
    type(ndarray) :: md_force
    type(tuple) :: return_value
    type(module_py) :: nprand, API_MoREST

    ierror = forpy_initialize()


    ierror = import_py(nprand, "numpy.random")
    ierror = import_py(API_MoREST,"API_MoREST")

    do i=1,10000
        if(i==1) then
            if_initial = 1
        else
            if_initial = 0
        endif

        ierror = call_py_noret(nprand, "random_sample()")
!        ierror = call_py(potential_energy, nprand, "random_sample()")
!        ierror = call_py(md_force, nprand, "rand()", 2, 3)
        current_md_step = current_md_step + 1

        write(*,*) current_md_step

    enddo

end program test_MoREST
