module api_morest

use, intrinsic :: iso_c_binding
implicit none

interface
    subroutine call_morest_its(if_initial, simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape) bind (c)
        use iso_c_binding
        integer(c_int64_t) :: if_initial, current_md_step, md_force_shape(2)
        real(c_double) :: simulation_temperature, potential_energy, md_force(md_force_shape(1),md_force_shape(2))
    end subroutine call_morest_its
end interface

public :: call_morest_its

integer(c_int64_t), public :: if_initial, current_md_step, md_force_shape(2)
real(c_double), public :: simulation_temperature, potential_energy
real(c_double), public, allocatable :: md_force(:,:)

end module api_morest
