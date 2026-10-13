#'
#' @title Matrix Cross-product using RcppEigen
#'
#' @description This function performs matrix cross-product using RcppEigen.
#'
#' \code{Eigen_crossprod} returns matrix cross-product of \code{A} and \code{B}, which is equivalent to (but faster than) \code{t(A) \%*\% B}.
#'
#' \code{Eigen_tcrossprod} returns matrix cross-product of \code{A} and \code{B}, which is equivalent to (but faster than) \code{A \%*\% t(B)}.
#'
#' @param A A numeric matrix. The first matrix to be multiplied.
#' @param B A numeric matrix. The second matrix to be multiplied.
#' @param n_cores An integer specifying the number of cores to use for parallel computation. Default is 4.
#'
#' @return A numeric matrix.
#' @examples
#' \donttest{
#' A <- matrix(1:4, 2, 2)
#' B <- matrix(4:1, 2, 2)
#' Eigen_crossprod(A, B)
#' Eigen_tcrossprod(A, B)
#'
#' # Test speed
#' A = matrix(rnorm(300*800),300,800)
#' B = matrix(rnorm(300*500),300,500)
#' bench::mark(t(A) %*% B, crossprod(A, B),
#'             Eigen_crossprod(A, B, n_cores = 1), Eigen_crossprod(A, B, n_cores = 4),
#'             check = TRUE)
#'
#' }

#' @export
#' @rdname Eigen_crossprod
Eigen_crossprod <- function(A, B, n_cores=4L){

  A <- as.matrix(A)
  B <- as.matrix(B)

  if (typeof(A) != "double") storage.mode(A) <- "double"
  if (typeof(B) != "double") storage.mode(B) <- "double"

  if (nrow(A) == nrow(B)) {
    return(Matrix_crossprod(A, B, n_cores))
  } else {
    stop("Dimension mismatch: nrow(x) != nrow(y)")
  }
}

#' @export
#' @rdname Eigen_crossprod
Eigen_tcrossprod <- function(A, B, n_cores=4L){

  A <- as.matrix(A)
  B <- as.matrix(B)

  if (typeof(A) != "double") storage.mode(A) <- "double"
  if (typeof(B) != "double") storage.mode(B) <- "double"

  if (ncol(A) == ncol(B)) {
    return(Matrix_tcrossprod(A, B, n_cores))
  } else {
    stop("Dimension mismatch: ncol(A) != ncol(B)")
  }
}
