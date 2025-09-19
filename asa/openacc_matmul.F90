! nvfortran -acc -O3 -Minfo=accel -o acc_matmul openacc_matmul.F90
program acc_matmul
  implicit none
  integer, parameter :: n=2048
  real, allocatable :: A(:,:), B(:,:), C(:,:)
  integer :: i,j,k
  real :: t0, t1

  allocate(A(n,n), B(n,n), C(n,n))
  A = 1.0; B = 2.0; C = 0.0

  call cpu_time(t0)
  !$acc data copyin(A,B) copy(C)
  !$acc parallel loop collapse(2) gang vector
  do i=1,n
     do j=1,n
        real :: sum
        sum = 0.0
        do k=1,n
           sum = sum + A(i,k)*B(k,j)
        end do
        C(i,j) = sum
     end do
  end do
  !$acc end data
  call cpu_time(t1)

  print *, "Done in", (t1-t0), "seconds"
end program
