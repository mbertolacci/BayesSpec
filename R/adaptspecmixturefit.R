.adaptspecmixturefit <- function(results, component_priors) {
  # Switch from zero-indexed samples to one-indexed
  results$categories <- results$categories + 1

  for (component in seq_len(results$n_components)) {
    results$components[[component]]$prior <- component_priors[[component]]
    results$components[[component]] <- .adaptspecfit(
      results$components[[component]]
    )
  }

  if (results$detrend && length(results$x_missing) > 0) {
    results$x_missing <- lapply(seq_len(length(results$x_missing)), function(i) {
      x_missing <- results$x_missing[[i]]
      if (is.null(x_missing)) return(x_missing)

      missing_indices <- results$missing_indices[[i]]
      x_base <- stats::predict(results$detrend_fits[[i]], data.frame(
        data0 = missing_indices
      ))
      x_missing + x_base
    })
  }

  class(results) <- 'adaptspecmixturefit'
  return(results)
}


#' @name adaptspecmixturefit
#' @title Methods for adaptspecmixturefit objects
#' @description These methods apply to the adaptspecmixturefit objects returned
#' by \code{\link{adaptspecx}}.
#' @param x \code{adaptspecmixturefit} object
#' @param top Text to show in the heading of the plot
#' @param ... Extra arguments. For \code{window}, these are passed to
#' \code{\link[coda]{window.mcmc}} to set the \code{thin} or \code{start}
#' parameters (for example)
#' @seealso
#' \code{\link{time_varying_mean_mean.adaptspecmixturefit}} for estimates of the
#' time-varying mean and spectral density.
NULL

#' @describeIn adaptspecmixturefit Method to modify the start/thinning of MCMC
#' samples, as per \code{\link[coda]{window.mcmc}}
#' @export
window.adaptspecmixturefit <- function(x, ...) {
  x$components <- lapply(x$components, window, ...)
  x$categories <- window(x$categories, ...)
  x$log_posterior <- window(x$log_posterior, ...)
  x$x_missing <- lapply(x$x_missing, function(x_missing) {
    if (is.null(x_missing)) {
      x_missing
    } else {
      window(x_missing, ...)
    }
  })
  x
}

#' @describeIn adaptspecmixturefit Returns the number of time periods in the
#' input
#' @export
ntimes.adaptspecmixturefit <- function(x, ...) {
  ntimes(x$components[[1]])
}

#' @describeIn adaptspecmixturefit Outputs MCMC diagnostics statistics to help
#' assess convergence. Calls \code{\link{diagnostics.adaptspecfit}}.
#' @export
diagnostics.adaptspecmixturefit <- function(x, ...) {
  cat(sprintf(
    'Tuning parameters: var_inflate = %f, prob_mm1 = %f\n',
    x$var_inflate,
    x$prob_mm1
  ))

  cat('Diagnostics for each component:\n')
  for (component in seq_len(x$n_components)) {
    cat(sprintf('============ Diagnostics for component %d:\n', component))
    diagnostics(x$components[[component]], ...)
  }
}

#' @describeIn adaptspecmixturefit MCMC diagnostic plots to assess convergence.
#' Calls \code{\link{diagnostics.adaptspecfit}}.
#' @export
diagnostic_plots.adaptspecmixturefit <- function(x, top = NULL, ...) {
  component_plots <- lapply(seq_len(x$n_components), function(component) {
    diagnostic_plots(
      x$components[[component]],
      top = sprintf('Component %d', component),
      ...
    )
  })
  do.call(
    gridExtra::grid.arrange,
    c(component_plots, list(ncol = 1, top = top))
  )
}

#' @describeIn adaptspecmixturefit Outputs warnings when MCMC diagnostics are
#' below outside of nominal threshold ranges. The user is cautioned that this is
#' subject to both false positive and false negatives; examining diagnostics
#' plots directly is advised.
#' Calls \code{\link{diagnostic_warnings.adaptspecfit}}.
#' @export
diagnostic_warnings.adaptspecmixturefit <- function(x, ...) {
  for (component in seq_len(x$n_components)) {
    diagnostic_warnings(
      x$components[[component]],
      prefix = sprintf('component %d, ', component),
      ...
    )
  }
}

#' @name time_varying_mean_mean.adaptspecmixturefit
#' @title Posterior estimates of the time varying mean and spectra from an
#' adaptspecmixturefit object
#'
#' @description These methods calculate posterior means or samples of the
#' time varying mean and spectra from fits performed using AdaptSPEC-X. These
#' functions can take a lot of time and memory, so consider thinning the input
#' \code{x} using \code{\link{window.adaptspecfit}} prior to calling, or
#' using the \code{time_step} argument.
#'
#' @param x \code{adaptspecmixturefit} object from \code{\link{adaptspecx}}
#' @param time_step Time varying quantity is calculated only at times
#' divisible by this number. Reduces the size of the output. Ignore if
#' \code{times} is provided.
#' @param times Times at which to calculate the quantity. Must be between
#' 1 and \code{ntimes(x)}, inclusive.
#' @param from Whether to calculate the quantity based on the mixture component
#' weights, or the component indicators for each time series.
#' @param n_frequencies Number of frequencies at which to evaluate the spectral
#' densities. Ignored if \code{frequencies} is set.
#' @param frequencies Frequencies at which to evaluate the spectral density.
#' Must be between 0 and 0.5, inclusive.
#' @param ... Ignored.
#' @examples
#' # Running adaptspecx with i.i.d. normal data
#' y <- matrix(rnorm(10000), ncol = 10)
#' design_matrix <- matrix(1, nrow = 10)
#' fit <- adaptspecx(
#'   3000,
#'   1000,
#'   y,
#'   design_matrix,
#'   n_components = 5,
#'   component_model = adaptspec_model(segment_means = TRUE)
#' )
#' fit_thin <- window(fit, thin = 10)
#' tvmm <- time_varying_mean_mean(fit_thin, time_step = 5)
#' tvsm <- time_varying_spectra_mean(
#'   fit_thin,
#'   n_frequencies = 128,
#'   time_step = 5
#' )
#' par(mfrow = c(1, 2))
#' matplot(attr(tvmm, 'times'), tvmm, type = 'l')
#' image(
#'   t(tvsm[, , 1]),
#'   x = attr(tvsm, 'times'),
#'   y = attr(tvsm, 'frequencies'),
#'   xlab = 'Time',
#'   ylab = 'Frequency',
#'   col = terrain.colors(50)
#' )
NULL


#' @describeIn time_varying_mean_mean.adaptspecmixturefit Posterior mean of the
#  time varying mean. Returns a matrix with one column per time sries.
#' @export
time_varying_mean_mean.adaptspecmixturefit <- function(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  from = c('probabilities', 'categories'),
  ...
) {
  from <- match.arg(from)
  n_iterations <- nrow(x$categories)
  n_times <- length(times)
  n_components <- length(x$components)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_times,
    n_components
  ))
  for (component in seq_len(n_components)) {
    component_samples[, , component] <- time_varying_mean_samples(
      x$components[[component]],
      times = times
    )
  }
  if (from == 'categories') {
    output <- .time_varying_mean_mixture_mean_categories(
      component_samples,
      x$categories
    )
  } else if (from == 'probabilities') {
    output <- .time_varying_mean_mixture_mean_probabilities(
      component_samples,
      component_probabilities(x)
    )
  }
  attr(output, 'times') <- times
  output
}

#' @describeIn time_varying_mean_mean.adaptspecmixturefit Posterior samples of
#' the time varying mean. Returns an array with dimensions (iteration, time,
#' time series)
#' @export
time_varying_mean_samples.adaptspecmixturefit <- function(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  from = c('probabilities'),
  ...
) {
  from <- match.arg(from)
  n_iterations <- nrow(x$categories)
  n_times <- length(times)
  n_components <- length(x$components)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_times,
    n_components
  ))
  for (component in seq_len(n_components)) {
    component_samples[, , component] <- time_varying_mean_samples(
      x$components[[component]],
      times = times
    )
  }
  if (from == 'probabilities') {
    output <- .time_varying_mean_mixture_samples_probabilities(
      component_samples,
      component_probabilities(x)
    )
  } else {
    stop('Value of from not supported')
  }
  attr(output, 'times') <- times
  output
}

#' @describeIn time_varying_mean_mean.adaptspecmixturefit Posterior mean
#' estimate of the time varying spectral density. Returns an array with
#' dimensions (frequency, time, time series).
#' @export
time_varying_spectra_mean.adaptspecmixturefit <- function(
  x,
  n_frequencies = 64,
  time_step = 1,
  from = c('probabilities', 'categories'),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  from <- match.arg(from)
  n_iterations <- nrow(x$categories)
  n_times <- length(times)
  n_components <- length(x$components)
  n_frequencies_actual <- length(frequencies)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_frequencies_actual,
    n_times,
    n_components
  ))
  for (component in seq_len(n_components)) {
    component_samples[, , , component] <- time_varying_spectra_samples(
      x$components[[component]],
      frequencies = frequencies,
      times = times
    )
  }
  if (from == 'categories') {
    output <- .time_varying_spectra_mixture_mean_categories(
      component_samples,
      x$categories
    )
  } else if (from == 'probabilities') {
    output <- .time_varying_spectra_mixture_mean_probabilities(
      component_samples,
      component_probabilities(x)
    )
  }
  attr(output, 'frequencies') <- frequencies
  attr(output, 'times') <- times
  output
}

#' @describeIn time_varying_mean_mean.adaptspecmixturefit Samples of the time
#' varying spectral density. Returns a four dimensional with dimensions
#' (iteration, frequency, time, time series).
#' @export
time_varying_spectra_samples.adaptspecmixturefit <- function(
  x,
  n_frequencies = 64,
  time_step = 1,
  from = c('probabilities', 'categories'),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  from <- match.arg(from)
  n_iterations <- nrow(x$categories)
  n_times <- length(times)
  n_components <- length(x$components)
  n_frequencies_actual <- length(frequencies)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_frequencies_actual,
    n_times,
    n_components
  ))
  for (component in seq_len(n_components)) {
    component_samples[, , , component] <- time_varying_spectra_samples(
      x$components[[component]],
      frequencies = frequencies,
      times = times
    )
  }
  if (from == 'probabilities') {
    output <- .time_varying_spectra_mixture_samples_probabilities(
      component_samples,
      component_probabilities(x)
    )
  }
  attr(output, 'frequencies') <- frequencies
  attr(output, 'times') <- times
  output
}

.merge_samples.adaptspecmixturefit <- function(x, fits) {  # nolint
  output <- .merge_mcmc_parts(fits[[1]], fits, c(
    'categories',
    'log_posterior'
  ))
  for (i in seq_along(fits[[1]]$components)) {
    output$components[[i]] <- merge_samples(
      lapply(fits, function(fit) fit$components[[i]])
    )
  }
  output$x_missing <- .merge_x_missing(fits)

  output
}
