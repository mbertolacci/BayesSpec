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
