#'
#' @title Predictions for cv.ECSLR Object
#'
#' @description \code{predict.cv.ECSLR} returns the predictions for a cv.ECSLR object.
#'
#' @method predict cv.ECSLR
#'
#' @param object An object of class cv.ECSLR.
#' @param newx New data for predictions.
#' @param new_costmatrix Each row of \code{new_costmatrix} represents the cost matrix of each sample in newx. The four columns are ("TN","FP","FN","TP") in order.
#' @param model_index The model for which to return the coefficients. Default is the ensemble.
#' @param type The type of predictions for binary response. Options are "prob" (default) and "class".
#' @param threshold.criteria The criteria for determining the threshold for classification. Options are "ID-cost" (default) and "balance". "ID-cost" uses instance-dependent cost-sensitive threshold, while "balance" uses a fixed threshold of 0.5.
#' @param ... Additional arguments for compatibility.
#'
#' @return The predictions for the cv.ECSLR object.
#'
#'
#' @seealso \code{\link{cv.ECSLR}}
#'
#'
#' @export
predict.cv.ECSLR <- function(object, newx, new_costmatrix, model_index = NULL, type = c("prob", "class")[1], threshold.criteria = c("ID-cost","balance")[1],...){
  ECSLR_object <- object$ECSLR_model_opt

  predict.ECSLR(ECSLR_object, newx, new_costmatrix, model_index, type, threshold.criteria,...)

}

