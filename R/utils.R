.extend_list <- function(...) {
  lists <- list(...)
  output <- lists[[1]]
  for (value in lists[2 : length(lists)]) {
    for (name in names(value)) {
      output[[name]] <- value[[name]]
    }
  }
  return(output)
}

#' @export
diagnostics <- function(x, ...) UseMethod('diagnostics', x)

#' @export
diagnostic_plots <- function(x, ...) UseMethod('diagnostic_plots', x)

#' @export
diagnostic_warnings <- function(x, ...) UseMethod('diagnostic_warnings', x)

#' @export
time_varying_spectra_samples <- function(x, ...) UseMethod('time_varying_spectra_samples', x)

#' @export
time_varying_spectra_mean <- function(x, ...) UseMethod('time_varying_spectra_mean', x)

#' @export
component_probabilities <- function(x, ...) UseMethod('component_probabilities', x)

.gcd <- function(a, b) {
  while (b != 0) {
    t <- b
    b <- a %% b
    a <- t
  }
  a
}

.lcm <- function(a, b) {
  as.integer((a * b) / .gcd(a, b))
}

.missing_indices <- function(data) {
  if (ncol(data) > 0) {
    lapply(1 : ncol(data), function(i) which(is.na(data[, i])))
  } else {
    list()
  }
}

.zero_index_missing_indices <- function(missing_indices) {
  lapply(missing_indices, `-`, 1)
}
