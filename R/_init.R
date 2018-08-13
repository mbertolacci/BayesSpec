#' @useDynLib BayesSpec
#' @importFrom Rcpp evalCpp
#' @importFrom futile.logger flog.debug

.onUnload <- function (libpath) {
  library.dynam.unload('BayesSpec', libpath)
}
