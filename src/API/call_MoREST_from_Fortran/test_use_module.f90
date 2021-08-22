program call_morest
  use, intrinsic :: iso_c_binding
  use api_morest
  implicit none

  integer(c_int64_t) :: i

  simulation_temperature = 798
  current_md_step = 1
  allocate(md_force(3,4))
  allocate(coordinate(3,4))
  md_force_shape = shape(md_force) !(/3,4/)
  coordinate_shape = shape(coordinate)
!  md_force = reshape((/1, 2, 3, 1, 3, 2, 3, 1, 2, 3, 2, 1/), md_force_shape)

  call random_seed()

  do i = 1,10000
    if (i == 1) then
      if_initial = 1
    else
      if_initial = 0
    endif

  call random_number(potential_energy)
  call random_number(md_force)
  call random_number(coordinate)

!  write(*,*) md_force
!  write(*,*) current_md_step

  call call_morest_bias_sampling(if_initial, simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape,&
                       coordinate, coordinate_shape)

!  write(*,*) md_force
!  write(*,*) current_md_step

  current_md_step = current_md_step + 1

  enddo

end program call_morest
