#' @name simulated_piecewise
#' @docType data
#' @aliases simulated_piecewise
#' @title Simulated Piecewise Time Series Dataset
#' @description This dataset is simulated from a piecewise autoregressive process (model (11), p. 1581, in Rosen et al. (2012)), see Examples.
#' @usage data(simulated_piecewise)
#' @format A univariate numeric vector with 1,024 observations.
#' @source Simulated
#' @references Rosen, O., Wood, S. and Stoffer, D. (2012). AdaptSPEC: Adaptive Spectral Estimation for Nonstationary Time Series. J. of the American Statistical Association, 107, 1575-1589
#' @keywords datasets
#' @examples
#' #Created using the following script:
#' set.seed(346)
#' phi_true <- matrix(list(),3,1)
#' phi_true[[1]] <- .9
#' phi_true[[2]] <- c(1.69, -.81)
#' phi_true[[3]] <- c(1.32, -.81)
#' sd_true <- rep(1,3)
#' x1 <- arima.sim(list(order=c(1,0,0), ar=phi_true[[1]]),512,sd=sd_true[1])
#' x2 <- arima.sim(list(order=c(2,0,0), ar=phi_true[[2]]),256,sd=sd_true[2])
#' x3 <- arima.sim(list(order=c(2,0,0), ar=phi_true[[3]]),256,sd=sd_true[3])
#' simulated_piecewise <- c(x1, x2, x3)
#' plot.ts(simulated_piecewise)
NULL
