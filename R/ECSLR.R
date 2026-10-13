#' @useDynLib ECSLR, .registration = TRUE
#' @title ECSLR: Ensemble Cost-sensitive Logistic Regression
#'
#' @description \code{ECSLR} performs the ECSLR algorithm for cost-sensitive binary classification tasks.
#' @param x A matrix of predictor variables.
#' @param y A vector of response variables.
#' @param cost_matrix A matrix representing costs. The four columns are ("TN","FP","FN","TP") in order.
#' @param M The number of individual models. Default is 10.
#' @param include_intercept Logical indicating whether to include an intercept term in the model. Default is TRUE.
#' @param lambda_sparsity The sparsity penalty parameter. Default is 0.01.
#' @param lambda_diversity The diversity penalty parameter. Default is 0.01.
#' @param tolerance The convergence tolerance for the optimization algorithm. Default is 1e-3.
#' @param max_iter The maximum number of iterations for the optimization algorithm. Default is 1e3.
#' @param x0 Optional initial values for the coefficients. Default is NULL.
#' @param start0 Logical indicating whether to include zero coefficients as the second starting values. Default is 'TRUE'.
#'
#' @return An object of class \code{ECSLR} containing the results of the ECSLR model, including coefficients and intercepts.
#' @examples
#' \donttest{
#' # Example usage of ECSLR function
#' # Load necessary libraries
#' library(ECSLR)
#'
#' # Simulate data
#' set.seed(2025)
#' n <- 100
#' p <- 100
#' x <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' beta <- c(abs(runif(p, 0, 1/2))*(-1)^rbinom(p, 1, 0.5))
#' prob <- sigmoid(x %**% beta)
#' y <- ifelse(prob>0.5,1,0)
#' cost_matrix <- matrix(0, nrow = n, ncol = 4)
#' cost_matrix[, 1] <- 0
#' cost_matrix[, 2] <- 1
#' cost_matrix[, 3] <- 5
#' cost_matrix[, 4] <- 0
#'
#' # Call the ECSLR function
#' ECSLR.obj <- ECSLR(x, y, cost_matrix,
#'                       M=10, include_intercept = TRUE,
#'                       lambda_sparsity = 0.01, lambda_diversity = 0.01)
#'
#' # Print the coefficients
#' coef(ECSLR.obj)
#'
#' y_pred <- predict(ECSLR.obj, newx = x, new_costmatrix = cost_matrix,
#'                   type = "class", threshold.criteria = "ID-cost")
#' table(y_pred, y)
#'
#' prob_pred <- predict(ECSLR.obj, newx = x, new_costmatrix = cost_matrix ,type = "prob")#'
#' plot(y, type = "p", pch = 16, col = "black", ylab = "Label", xlab = "Sample Index",
#'      main = "True (Black) vs Predicted Prob. (Red)")
#' points(prob_pred, pch = 1, col = "red")
#' }
#' @seealso \code{\link{coef.ECSLR}}, \code{\link{predict.ECSLR}}
#' @author Bing Yang, Stefan Van Aelst, Tim Verdonck \cr Maintainer: Bing Yang (\email{bing.yang@@kuleuven.be})
#' @references
#' Yang, B., Van Aelst, S., & Verdonck, T. (2025). Diverse ensemble cost-sensitive logistic regression. \emph{European Journal of Operational Research}. \url{https://doi.org/10.1016/j.ejor.2025.07.028}
#'
#' @export
ECSLR <- function(x, y, cost_matrix,
                  M = 10, include_intercept = TRUE,
                  lambda_sparsity = 0.01,
                  lambda_diversity = 0.01,
                  tolerance = 1e-3, max_iter = 1e3, x0 = NULL, start0 = TRUE){
  # Check if the input data is a matrix or data.frame
  if (!is.matrix(x) && !is.data.frame(x)) {
    stop("x must be a matrix or data frame.")
  }

  # Check if the response variable is a binary variable or factor
  if (!is.numeric(y) && !is.factor(y) && !is.character(y) && !is.logical(y)) {
    stop("y must be a numeric, factor, character, or logical vector.")
  }

  if (is.factor(y)) {
    y <- as.numeric(as.character(y))
  }else{
    y <- as.numeric(y)
  }

  # Check if the response variable is binary
  if (length(unique(y)) != 2) {
    stop("y must be a binary variable.")
  }

  # Check if the cost matrix is a matrix with 4 columns
  if (!is.matrix(cost_matrix) || ncol(cost_matrix) != 4) {
    stop("cost_matrix must be a matrix with 4 columns.")
  }

  # Check if the number of rows in x matches the length of y
  if (nrow(x) != length(y)) {
    stop("The number of rows in x must match the length of y.")
  }

  # Check if the number of rows in cost_matrix matches the number of rows in x
  if (nrow(cost_matrix) != nrow(x)) {
    stop("The number of rows in cost_matrix must match the number of rows in x.")
  }

  x <- as.matrix(x)

  ECSLR_opt <- ECSLR_multi_initial(x, y, cost_matrix,
                                   M, include_intercept,
                                   alpha_s = 1,
                                   lambda_sparsity,
                                   lambda_diversity,
                                   tolerance, max_iter, model = "ECSLR", x0, start0)


  row.names(ECSLR_opt$betas_scaled) <- colnames(x)
  names(ECSLR_opt$betas_est) <- colnames(x)

  row.names(ECSLR_opt$intercept_scaled) <- "(Intercept)"
  names(ECSLR_opt$intercept_est) <- "(Intercept)"

  ECSLR_opt$call <- match.call()

  return(ECSLR_opt)
}
