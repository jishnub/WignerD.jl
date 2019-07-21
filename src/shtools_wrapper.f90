module shtools_wrapper

use, intrinsic :: iso_c_binding
use SHTOOLS
implicit none

contains 

subroutine wigner3j_wrapper(w3j,len,j2, j3, m1, m2, m3, exitstatus) bind(C, name="wigner3j_wrapper")
    integer(kind=C_INT32_T), intent(in)  :: len,j2,j3,m1,m2,m3
    real(kind=C_DOUBLE), intent(out) :: w3j(len)
    integer(kind=C_INT32_T) :: jmin,jmax
    integer(kind=C_INT32_T), intent(out), optional :: exitstatus

    call Wigner3j(w3j, jmin, jmax, j2, j3, m1, m2, m3, exitstatus=exitstatus)
end subroutine wigner3j_wrapper

end module shtools_wrapper