# Do ECSLR from multi initial values
ECSLR_multi_initial <- function(x, y, cost_matrix,
                                M, include_intercept,
                                alpha_s,
                                lambda_sparsity,
                                lambda_diversity,
                                tolerance, max_iter, model, x0 = NULL, start0 = TRUE){
  p <- ncol(x)

  # start from given start value
  if(is.null(x0)){
    ECSLR_model_start_previous <- list(objective=NA)
  }else{
    ECSLR_model_start_previous <- ECSLR_Compute_Coef(x, y, cost_matrix,
                                                     M, include_intercept,
                                                     alpha_s,
                                                     lambda_sparsity,
                                                     lambda_diversity,
                                                     tolerance, max_iter, model, x0=x0)
    ECSLR_model_start_previous$opt_start <- "x0"
    # cat("start_previous:",ECSLR_model_start_previous$objective,"\n\n")
  }

  if(start0){
    # start from given 0
    ECSLR_model_start_0 <- ECSLR_Compute_Coef(x, y, cost_matrix,
                                              M, include_intercept,
                                              alpha_s,
                                              lambda_sparsity,
                                              lambda_diversity,
                                              tolerance, max_iter, model, x0=NULL) #NULL means start from 0
    ECSLR_model_start_0$opt_start <- "0"
    # cat("start_0:",ECSLR_model_start_0$objective,"\n\n")
  }else{
    ECSLR_model_start_0 <- list(objective=NA)
  }

  models <- list(ECSLR_model_start_previous, ECSLR_model_start_0)

  ECSLR_opt <- models[[which.min(sapply(models, function(x) x$objective))]] #sapply

  class(ECSLR_opt) <- c("ECSLR", "list")

  return(ECSLR_opt)
}


instance_dependent_cost_threshold <- function(cost_matrix){
  (cost_matrix[, 2] - cost_matrix[, 1]) /(cost_matrix[, 2] - cost_matrix[, 1] +
                                                  cost_matrix[, 3] - cost_matrix[, 4])
}

# Compute_coef_cost <- function(y,cost_matrix){
#   a <- y * (cost_matrix[,4] - cost_matrix[,3]) + (1-y) * (cost_matrix[,2]-cost_matrix[,1])
#   b <- y * cost_matrix[,3] + (1-y)*cost_matrix[,1]
#   return(list(a=a,b=b))
# }

Compute_aec_cost <- function(intercept1, betas1, x, y, cost_matrix, model){
  scores <- sigmoid(intercept1 + Matrix_Mult(as.matrix(x), betas1))
  opt_threshold <- instance_dependent_cost_threshold(cost_matrix)
  pre.class <- ifelse(scores>opt_threshold,1,0)

  cost <- Compute_AEC(pre.class, y, cost_matrix=cost_matrix) #using cost as cross validation criteria
  AEC <- Compute_AEC(scores, y, cost_matrix=cost_matrix)
  return(list(cost=cost,AEC=AEC))
}
