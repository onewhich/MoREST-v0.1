program call_python
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int64_t) :: i,j
  integer(c_int64_t) :: x_shape(2)
  integer(c_int64_t), parameter :: n_atom=4
  real(c_double) :: x(3,n_atom)

  interface
    subroutine add_one(x_c, n) bind (c)
        use iso_c_binding
        integer(c_int64_t) :: n(2)
        real(c_double) :: x_c(3,n(2))
    end subroutine add_one
  end interface
  
  do i=1,3
    do j=1,n_atom
      x(i,j)=i+j*3
    enddo
  enddo

  do i=1,3
    do j=1,n_atom
      write(*,*) x(i,j)
    enddo
  enddo
  print *, x

  x_shape = shape(x)
  call add_one(x, x_shape)
  print *, x

  print *, x_shape

end program call_python
