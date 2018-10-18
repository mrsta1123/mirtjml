#' Exploratory item factor analysis for high dimensinal dichotomous responses data
#'
#' @param response Matrix contains 0/1 responses for N rows of examinees and J columns of items.
#' @param nonmis_ind Matrix contains 0/1 elements with 1 stands for non-missing response in the coresponding position.
#' @param K Integer for specifying the latent dimension in exploratory item factor analysis.
#' @param theta0 The initial value for person parameters. The default is set to be null and a 
#' default value based on singular value decomposition of resposne data will be computated.
#' @param A0 The initial value for item parameters. The default is set to be null and a 
#' default value based on singular value decomposition of resposne data will be computated.
#' @param d0 The initial value of the intercept parameters. The default is set to be null and a 
#' default value based on singular value decomposition of resposne data will be computated.
#' @param C The constraint constant. The default is set be to null and a default value is computed 
#' based on the given latent dimension K.
#' @param tol The tolerance parameter. The default value is 1e-4.
#' @param trans_ogive Is the result loading parameters transformed to ogive model, default is TRUE.
#' @param print_proc Print the precision during the esitmation procedure. The default is TRUE
#' 
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{theta_hat}{The estimated person parameters matrix.}
#'   \item{A_hat}{The estimated loading parameters matrix}
#'   \item{d_hat}{The estimated value of intercept parameters.}
#' }
#' @references 
#' Joint Maximum Likelihood Estimation for High-dimensional Exploratory Item Factor Analysis
#' 
#' @importFrom GPArotation GPFoblq
#' @export mirtjml
mirtjml <- function(response, nonmis_ind, K, theta0 = NULL, A0 = NULL, d0 = NULL, C = NULL, 
                    tol = 1e-4, trans_ogive = TRUE, print_proc = TRUE){
  N <- nrow(response)
  J <- ncol(response)
  if(is.null(theta0) || is.null(A0) || is.null(d0)){
    t1 <- Sys.time()
    if(print_proc){
      cat("\n", "Initializing... finding good starting point.\n")
    }
    initial_value = svd_start(response, nonmis_ind, K)
    t2 <- Sys.time()
  }
  if(is.null(theta0)){
    theta0 <- initial_value$theta0
  }
  if(is.null(A0)){
    A0 <- initial_value$A0
  }
  if(is.null(d0)){
    d0 <- initial_value$d0
  }
  if(is.null(C)){
    C = 5*sqrt(K)
  }
  res <- alter_mini_CJMLE_cpp(response, nonmis_ind, 
                              cbind(rep(1,N),theta0),cbind(d0,A0), C, tol, print_proc)
  res_standard <- standardization_cjmle(res$theta[,2:(K+1)], res$A[,2:(K+1)], res$A[,1])
  if(K > 1){
    temp <- GPFoblq(res_standard$A1, method = "geomin")
    A_rotated <- temp$loadings
    rotation_M <- temp$Th
    theta_rotated <- res_standard$theta1 %*% rotation_M
  } else{
    A_rotated <- res_standard$A1
    theta_rotated <- res_standard$theta1
  }
  if(trans_ogive){
    A_rotated <- A_rotated / 1.702
  }
  t3 <- Sys.time()
  if(print_proc){
    cat("\n\n", "Precision reached! Possible solution is found. \n")
    cat("Time spent:\n")
    cat("Find start point: ", as.numeric(t2-t1)," | ", "Optimization: ", as.numeric(t3-t2),"\n")
  }
  return(list("theta_hat" = theta_rotated,
              "A_hat" = A_rotated,
              "d_hat" = res_standard$d1))
}


