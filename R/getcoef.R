#' Extract coefficients from a ECSLR object
#'
#' @method coef ECSLR
#'
#' @param object Fitted ECSLR model object.
#' @param model_index The model for which to return the coefficients. Default is the ensemble coefficients.
#' @param ... Additional arguments for compatibility.
#'
#' @return The coefficients for the \code{\link{ECSLR}} object.
#'
#'
#' @seealso \code{\link{ECSLR}}
#'
#' @export
coef.ECSLR <- function(object, model_index = NULL, ...){

  # Check input data
  if(!any(class(object) %in% "ECSLR"))
    stop("The object should be of class \"ECSLR\"")

  coef.final <- c(object$intercept_est, object$betas_est)
  if(object$M==1 | is.null(model_index)){
    return(coef.final)
  }else{
    if(!(model_index %in% 1:object$M))
      stop("The group index is not valid.")
    return(c(object$intercept_scaled[model_index],object$betas_scaled[,model_index]))
  }
}


#' Extract coefficients from a cv.ECSLR object
#'
#' @method coef cv.ECSLR
#'
#' @param object Fitted cv.ECSLR model object.
#' @param model_index The model for which to return the coefficients. Default is the ensemble coefficients.
#' @param ... Additional arguments for compatibility.
#'
#' @return The coefficients for the \code{\link{cv.ECSLR}} object.
#'
#' @seealso \code{\link{cv.ECSLR}}

#' @export
coef.cv.ECSLR <- function(object, model_index = NULL, ...){

  # Check input data
  if(!any(class(object) %in% "cv.ECSLR"))
    stop("The object should be of class \"cv.ECSLR\"")

  coef.final <- c(object$ECSLR_model_opt$intercept_est, object$ECSLR_model_opt$betas_est)
  if(object$M==1 | is.null(model_index)){
    return(coef.final)
  }else{
      if(!(model_index %in% 1:object$M))
        stop("The group index is not valid.")
      return(c(object$ECSLR_model_opt$intercept_scaled[model_index],object$ECSLR_model_opt$betas_scaled[,model_index]))
  }
}
