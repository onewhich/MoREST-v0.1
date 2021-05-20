program call_morest
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int64_t) :: if_initial, current_md_step, md_force_shape(2), i
  real(c_double) :: simulation_temperature, potential_energy, md_force(3,4)

  interface
    subroutine call_morest_its(if_initial, simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape) bind (c)
        use iso_c_binding
        integer(c_int64_t) :: if_initial, current_md_step, md_force_shape(2)
        real(c_double) :: simulation_temperature, potential_energy, md_force(3,4)
    end subroutine call_morest_its
  end interface

  simulation_temperature = 798
  md_force_shape = (/3,4/)
!  md_force = reshape((/1, 2, 3, 1, 3, 2, 3, 1, 2, 3, 2, 1/), md_force_shape)

  call random_seed()

  do i = 1,1000
    if (i == 1) then
      if_initial = 1
    else
      if_initial = 0
    endif

  call random_number(potential_energy)
  call random_number(md_force)
  current_md_step = i-1
  call call_morest_its(if_initial, simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape)

  enddo

end program call_morest
