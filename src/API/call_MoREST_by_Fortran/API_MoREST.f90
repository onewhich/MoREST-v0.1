module api_morest

use, intrinsic :: iso_c_binding
implicit none

interface
    subroutine call_morest_bias_sampling(simulation_temperature, potential_energy,&
                       current_md_step, md_force, md_force_shape,&
                       coordinate, coordinate_shape) bind (c)
        use iso_c_binding
        integer(c_int64_t) :: current_md_step, md_force_shape(2), coordinate_shape(2)
        real(c_double) :: simulation_temperature, potential_energy,&
                    md_force(md_force_shape(1),md_force_shape(2)),&
                    coordinate(coordinate_shape(1),coordinate_shape(2))
    end subroutine call_morest_bias_sampling
end interface

public :: call_morest_bias_sampling

integer(c_int64_t), public :: current_md_step, md_force_shape(2), coordinate_shape(2)
real(c_double), public :: simulation_temperature, potential_energy
real(c_double), public, allocatable :: md_force(:,:), coordinate(:,:)

end module api_morest
