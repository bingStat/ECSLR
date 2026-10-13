#' @title Matrix Multiplication using RcppEigen
#'
#' @description This function performs matrix multiplication using RcppEigen.
#' \code{\%**\%} returns multiplication of \code{A} and \code{B}, which is equivalent to  (but faster than) \code{\%*\%}.
#'
#' @param A A numeric matrix. The first matrix to be multiplied.
#' @param B A numeric matrix. The second matrix to be multiplied.
#' @return A numeric matrix.
#' @examples
#' \donttest{
#' A <- matrix(1:4, 2, 2)
#' B <- matrix(4:1, 2, 2)
#' A %**% B
#'
#' # Test speed of matrix multiplication with different methods
#' A = matrix(rnorm(200*300),200,300)
#' B = matrix(rnorm(300*400),300,400)
#' bench::mark(A %*% B, A %**% B, check = TRUE)
#' }

#' @export
#' @rdname matrix_multiplication
# Matrix multiplication using RcppEigen
"%**%" <- function(A, B){

  A <- as.matrix(A)
  B <- as.matrix(B)

  if (typeof(A) != "double") storage.mode(A) <- "double"
  if (typeof(B) != "double") storage.mode(B) <- "double"

  if (ncol(A) != nrow(B)) {
    stop("Dimension mismatch: ncol(A) != nrow(B)")
  }

  return(Matrix_Mult(A, B))

}
