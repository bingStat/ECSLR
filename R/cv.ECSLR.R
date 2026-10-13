#'
#' @importFrom foreach foreach %dopar% %:%
#' @importFrom parallel makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom caret createMultiFolds
#' @importFrom stats coef quantile sd
#' @importFrom utils capture.output
#'
#' @title Cross Validation - Ensemble Cost-sensitive Logistic Regression
#'
#' @description \code{cv.ECSLR} performs the CV procedure for ensemble cost-sensitive logistic regression.
#'
#' @param x Design matrix.
#' @param y Response vector.
#' @param cost_matrix The cost matrix. Row i represents the cost-matrix of sample i in \code{x}. The four columns are in the form "TN","FP","FN","TP".
#' @param M Number of models in the ensemble. The default value is 10. \code{M} can also be a vector, in this case, a cross-validation will be performed for the candidated values in \code{M}
#' @param include_intercept Boolean variable to determine if there is intercept (default is TRUE) or not.
#' @param n_lambda_sparsity Number of candidates for the sparsity penalty parameter. Default is 20.
#' @param n_lambda_diversity Number of candidates for the sparsity penalty parameter. Default is 20.
#' @param tolerance Convergence criteria for the coefficients. Default is 1e-3.
#' @param max_iter Maximum number of iterations in the algorithm. Default is 1e3.
#' @param n_folds Number of cross-validation folds. Default is 3.
#' @param criteria Character string specifying the criteria for tuning parameters in the cross-valiation process. Can be "cost"(by default) or "AEC".
#' @param start0 Logical indicating whether to include zero coefficients as the second starting values in the cross-valiadation process. Default is TRUE.
#' @param n_threads Integer specifying the number of threads for parallel computation during cross validation. Default is 3.
#' @param verbose Logical indicating whether to print the progress of the cross-validation. Default is TRUE.
#' @return An object containing the results of the cross-validated ECSLR model, including
#'         selected parameters, coefficients, and cross-validation metrics.
#'
#' @examples
#' \donttest{
#' # Example usage of cv.ECSLR function
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
#' # Call the cv.ECSLR function
#' cv.ECSLR.obj <- cv.ECSLR(x, y, cost_matrix, M=10, include_intercept=TRUE,
#'                          n_lambda_sparsity=20, n_lambda_diversity=20, n_folds=3,
#'                          n_threads=1, verbose = TRUE)
#' # Print the coefficients
#' coef(cv.ECSLR.obj)
#'
#' y_pred <- predict(cv.ECSLR.obj, newx = x, new_costmatrix = cost_matrix,
#'                   type = "class", threshold.criteria = "ID-cost")
#' table(y_pred, y)
#'
#' prob_pred <- predict(cv.ECSLR.obj, newx = x, new_costmatrix = cost_matrix ,type = "prob")
#' plot(y, type = "p", pch = 16, col = "black", ylab = "Label", xlab = "Sample Index",
#'      main = "True (Black) vs Predicted Prob. (Red)")
#' points(prob_pred, pch = 1, col = "red")
#' }
#' @seealso \code{\link{coef.cv.ECSLR}}, \code{\link{predict.cv.ECSLR}}
#'
#' @author Bing Yang, Stefan Van Aelst, Tim Verdonck \cr Maintainer: Bing Yang (\email{bing.yang@@kuleuven.be})
#' @references
#' Yang, B., Van Aelst, S., & Verdonck, T. (2025). Diverse ensemble cost-sensitive logistic regression. \emph{European Journal of Operational Research}. \url{https://doi.org/10.1016/j.ejor.2025.07.028}
#' @export
cv.ECSLR <- function(x, y, cost_matrix, M=10, include_intercept=TRUE,
                     n_lambda_sparsity=20, n_lambda_diversity=20, n_folds=3,
                     tolerance=1e-3, max_iter=1e3, criteria = "cost", start0 = TRUE, n_threads=3, verbose = TRUE){

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

  if(verbose){
    cv.ECSLR.obj <- Compute_CV_Betas_alternating(x, y, x_categorical=NULL, categorical_variables=character(0),
                                                 Ms = M, include_intercept = include_intercept, alpha_s = 1,
                                                 n_lambda_sparsity, n_lambda_diversity, n_folds,
                                                 tolerance, max_iter, model="ECSLR", cost_matrix = cost_matrix, criteria = criteria, start0 = start0, n_threads=n_threads)

  }else{
    invisible(capture.output({
      cv.ECSLR.obj <- Compute_CV_Betas_alternating(x, y, x_categorical=NULL, categorical_variables=character(0),
                                                   Ms = M, include_intercept = include_intercept, alpha_s = 1,
                                                   n_lambda_sparsity, n_lambda_diversity, n_folds,
                                                   tolerance, max_iter, model="ECSLR", cost_matrix = cost_matrix, criteria = criteria, start0 = start0, n_threads=n_threads)

    }))
  }

  row.names(cv.ECSLR.obj$ECSLR_model_opt$betas_scaled) <- colnames(x)
  names(cv.ECSLR.obj$ECSLR_model_opt$betas_est) <- colnames(x)

  row.names(cv.ECSLR.obj$ECSLR_model_opt$intercept_scaled) <- "(Intercept)"
  names(cv.ECSLR.obj$ECSLR_model_opt$intercept_est) <- "(Intercept)"

  cv.ECSLR.obj$call <- match.call()

  return(cv.ECSLR.obj)
}
