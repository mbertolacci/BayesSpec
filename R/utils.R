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

#' Generic for the number of time periods in the input
#'
#' Generic function to return the numbers of time periods in the input for
#' either an AdaptSPEC or AdaptSPEC-X object. See \code{\link{adaptspecfit}}
#' or \code{\link{adaptspecmixturefit}} for the class-specific methods.
#'
#' @param x The object to return the number of times for
#' @param ... Extra argument, possibly used by class-specific mixtures.
#' @export
ntimes <- function(x, ...) {
  UseMethod('ntimes', x)
}

#' Generics for diagnostic methods
#'
#' These generics perform various diagnostic routines for AdaptSPEC and
#' AdaptSPEC-X outputs. See \code{\link{adaptspecfit}}
#' or \code{\link{adaptspecmixturefit}} for the class-specific methods.
#'
#' @param x The object to perform the diagnostics for.
#' @param ... Extra argument, possibly used by class-specific mixtures.
#' @export
diagnostics <- function(x, ...) UseMethod('diagnostics', x)

#' @describeIn diagnostics Create diagnostic plots.
#' @export
diagnostic_plots <- function(x, ...) UseMethod('diagnostic_plots', x)

#' @describeIn diagnostics Give diagnostic warnings.
#' @export
diagnostic_warnings <- function(x, ...) UseMethod('diagnostic_warnings', x)

#' Generics for calculating the time varying mean and spectrum
#'
#' These generics perform calculate quantities related to the time varying mean
#' and spectra for fits made with AdaptSPEC and AdaptSPEC-X outputs. See
#' See \code{\link{time_varying_mean_mean.adaptspecfit}} or
#' \code{\link{time_varying_mean_mean.adaptspecmixturefit}} for class-specific
#' methods.
#'
#' @param x The object to calculate the quantity for.
#' @param ... Extra argument, possibly used by class-specific mixtures.
#' @export
time_varying_mean_mean <- function(x, ...) {
  UseMethod('time_varying_mean_mean', x)
}

#' @describeIn time_varying_mean_mean Posterior samples from the time varying
#' mean.
#' @export
time_varying_mean_samples <- function(x, ...) {
  UseMethod('time_varying_mean_samples', x)
}

#' @describeIn time_varying_mean_mean Posterior mean of the time varying
#' spectra.
#' @export
time_varying_spectra_mean <- function(x, ...) {
  UseMethod('time_varying_spectra_mean', x)
}

#' @describeIn time_varying_mean_mean Posterior samples from the time varying
#' spectra.
#' @export
time_varying_spectra_samples <- function(x, ...) {
  UseMethod('time_varying_spectra_samples', x)
}

.merge_samples <- function(x, ...) {
  UseMethod('.merge_samples', x)
}

#' Merge samples from AdaptSPEC or AdaptSPEC-X
#'
#' This function merges samples from different runs of AdaptSPEC or AdaptSPEC-X
#' into a single object. This is used to pool samples from MCMC runs made in
#' parallel.
#'
#' @param values List of objects to merge, either from \code{\link{adaptspec}}
#' or \code{\link{adaptspecx}}
#' @return The merged object
#' @export
merge_samples <- function(values) {
  .merge_samples(values[[1]], values)
}

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
  lapply(seq_len(ncol(data)), function(i) which(is.na(data[, i])))
}

.zero_index_missing_indices <- function(missing_indices) {
  lapply(missing_indices, `-`, 1)
}

.prepare_data <- function(data, detrend) {
  data <- as.matrix(data)
  detrend_fits <- NULL
  if (detrend && ncol(data) > 0) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    data0 <- seq_len(nrow(data))  # nolint
    detrend_fits <- list()
    for (series in seq_len(ncol(data))) {
      detrend_fits[[series]] <- stats::lm(
        data[, series] ~ data0,
        na.action = stats::na.exclude
      )
      data[, series] <- stats::residuals(detrend_fits[[series]])
    }
  }

  list(
    data = data,
    detrend_fits = detrend_fits,
    missing_indices = .missing_indices(data)
  )
}

.x_missing_start <- function(start, missing_indices) {
  if (is.null(start$x_missing)) {
    start$x_missing <- lapply(missing_indices, function(x) {
      stats::rnorm(length(x))
    })
  }
  start
}

.validate_x_missing_start <- function(start, missing_indices) {
  for (i in seq_len(length(missing_indices))) {
    stopifnot(length(missing_indices[[i]]) == length(start$x_missing[[i]]))
  }
}

.merge_mcmc <- function(values) {
  if (is.null(values[[1]])) {
    return(NULL)
  }
  if (is.null(dim(values[[1]]))) {
    how <- c
  } else if (length(dim(values[[1]])) < 3) {
    how <- rbind
  } else {
    how <- function(...) abind::abind(..., along = 1)
  }
  output <- do.call(how, values)
  if (is.null(dim(output)) || length(dim(output)) < 3) {
    class(output) <- 'mcmc'
  } else {
    class(output) <- 'mcmca'
  }
  n_output <- NULL
  if (is.null(dim(output))) {
    n_output <- length(output)
  } else {
    n_output <- dim(output)[1]
  }
  attr(output, 'mcpar') <- c(1, n_output, 1)
  output
}

.merge_mcmc_parts <- function(start, values, parts) {
  output <- start
  for (part in parts) {
    output[[part]] <- .merge_mcmc(lapply(values, getElement, part))
  }
  output
}

.merge_x_missing <- function(fits) {
  lapply(seq_len(length(fits[[1]]$x_missing)), function(i) {
    if (is.null(fits[[1]]$x_missing[[i]])) NULL
    else {
      values <- lapply(fits, function(fit) fit$x_missing[[i]])
      .merge_mcmc(values)
    }
  })
}
