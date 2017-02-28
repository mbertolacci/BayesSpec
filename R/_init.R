#' @useDynLib BayesSpec
#' @importFrom Rcpp evalCpp

.onUnload <- function (libpath) {
    library.dynam.unload('BayesSpec', libpath)
}
