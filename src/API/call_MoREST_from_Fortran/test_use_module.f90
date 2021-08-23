program call_morest
  use, intrinsic :: iso_c_binding
  use api_morest
  implicit none

  integer(c_int64_t) :: i

  simulation_temperature = 798
!  current_md_step = 1
  allocate(md_force(3,2))
  allocate(coordinate(3,2))
  md_force_shape = shape(md_force) !(/3,4/)
  coordinate_shape = shape(coordinate)
  md_force = reshape((/1, 2, 3, 1, 3, 2/), md_force_shape) !, 3, 1, 2, 3, 2, 1/), md_force_shape)
  coordinate = reshape((/3, 1, 2, 3, 2, 1/), coordinate_shape)
  potential_energy = 1

  call random_seed()

!  if_initial = 1
  do i = 1,2
    current_md_step = i
!    if (i == 1) then
!      if_initial = 0
!    else
!      if_initial = 0
!    endif

!  write(*,*) if_initial

!  call random_number(potential_energy)
!  call random_number(md_force)
!  call random_number(coordinate)

!  write(*,*) md_force
!    write(*,*) current_md_step

    call call_morest_bias_sampling(simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape,&
                       coordinate, coordinate_shape)

    potential_energy = potential_energy + 1
!  write(*,*) md_force
!  write(*,*) current_md_step

!    if_initial = 0
!    current_md_step = current_md_step + 1

  enddo

end program call_morest
