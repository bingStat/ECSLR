#'
#' @title Predictions for ECSLR Object
#'
#' @description \code{predict.ECSLR} returns the predictions for a ECSLR object.
#'
#' @method predict ECSLR
#'
#' @param object An object of class ECSLR.
#' @param newx New data for predictions.
#' @param new_costmatrix Each row of \code{new_costmatrix} represents the cost matrix of each sample in newx. The four columns are ("TN","FP","FN","TP") in order.
#' @param model_index The model for which to return the coefficients. Default is the ensemble.
#' @param type The type of predictions for binary response. Options are "prob" (default) and "class".
#' @param threshold.criteria The criteria for determining the threshold for classification. Options are "ID-cost" (default) and "balance". "ID-cost" uses instance-dependent cost-sensitive threshold, while "balance" uses a fixed threshold of 0.5.
#' @param ... Additional arguments for compatibility.
#'
#' @return The predictions for the ECSLR object.
#'
#'
#' @seealso \code{\link{ECSLR}}
#'
#'
#' @export
predict.ECSLR <- function(object, newx, new_costmatrix, model_index = NULL, type = c("class", "prob")[1], threshold.criteria = c("ID-cost","balance")[1],...){

  # Check input data
  if(!any(class(object) %in% "ECSLR"))
    stop("The object should be of class \"ECSLR\"")

  if(!(type %in% c("prob", "class")))
    stop("The variable \"type\" must be one of: \"prob\", or \"class\".")

  ecslr.coef <- coef(object, model_index=model_index)

  pre.prop.newx <- sigmoid(ecslr.coef[1] + newx %*% ecslr.coef[-1])

  if(type=="prob"){
    return(pre.prop.newx)
  }else if(type=="class"){
    if(threshold.criteria=="balance"){
      threshold <- 0.5
    }else if(threshold.criteria=="ID-cost"){
      if(nrow(new_costmatrix)!=nrow(newx)){
        stop("The row of \"new_costmatrix\" must match the row of \"newx\".")
      }
      if(ncol(new_costmatrix)!=4){
        stop("The column of \"new_costmatrix\" must be 4, corresponding to (\"TN\",\"FP\",\"FN\",\"TP\") in order.")
      }
      #   Instance-dependent cost-sensitive threshold
      threshold <- instance_dependent_cost_threshold(new_costmatrix)
    }
    pre.class.newx <- ifelse(pre.prop.newx>threshold,1,0)
    return(pre.class.newx)
  }
}
