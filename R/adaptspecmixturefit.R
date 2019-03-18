adaptspecmixturefit <- function(results, component_priors) {
  # Switch from zero-indexed samples to one-indexed
  results$categories <- results$categories + 1

  for (component in 1 : results$n_components) {
    results$components[[component]]$prior <- component_priors[[component]]
    results$components[[component]] <- adaptspecfit(
      results$components[[component]]
    )
  }

  if (results$detrend && length(results$x_missing) > 0) {
    results$x_missing <- lapply(1 : length(results$x_missing), function(i) {
      x_missing <- results$x_missing[[i]]
      if (is.null(x_missing)) return(x_missing)

      missing_indices <- results$missing_indices[[i]]
      x_base <- predict(results$detrend_fits[[i]], data.frame(
        data0 = missing_indices
      ))
      x_missing + x_base
    })
  }

  class(results) <- 'adaptspecmixturefit'
  return(results)
}

#' @export
window.adaptspecmixturefit <- function(fit, ...) {
  fit$components <- lapply(fit$components, window, ...)
  fit$categories <- window(fit$categories, ...)
  fit$log_posterior <- window(fit$log_posterior, ...)
  fit$x_missing <- lapply(fit$x_missing, function(x_missing) {
    if (is.null(x_missing)) {
      x_missing
    } else {
      window(x_missing, ...)
    }
  })
  fit
}

#' @export
ntimes.adaptspecmixturefit <- function(fit) {
  ntimes(fit$components[[1]])
}

#' @export
diagnostics.adaptspecmixturefit <- function(fit, ...) {
  cat(sprintf(
    'Tuning parameters: var_inflate = %f, prob_mm1 = %f\n',
    fit$var_inflate,
    fit$prob_mm1
  ))

  cat('Diagnostics for each component:\n')
  for (component in 1 : fit$n_components) {
    cat(sprintf('============ Diagnostics for component %d:\n', component))
    diagnostics(fit$components[[component]], ...)
  }
}

#' @export
diagnostic_plots.adaptspecmixturefit <- function(fit, top = NULL, ...) {
  component_plots <- lapply(1 : fit$n_components, function(component) {
    diagnostic_plots(
      fit$components[[component]],
      top = sprintf('Component %d', component),
      ...
    )
  })
  do.call(
    gridExtra::grid.arrange,
    c(component_plots, list(ncol = 1, top = top))
  )
}

#' @export
diagnostic_warnings.adaptspecmixturefit <- function(fit, ...) {
  for (component in 1 : fit$n_components) {
    diagnostic_warnings(
      fit$components[[component]],
      prefix = sprintf('component %d, ', component),
      ...
    )
  }
}

#' @export
time_varying_mean_mean.adaptspecmixturefit <- function(
  fit,
  time_step = 1,
  times = seq(1, ntimes(fit), by = time_step),
  from = c('probabilities', 'categories')
) {
  from <- match.arg(from)
  n_iterations <- nrow(fit$categories)
  n_times <- length(times)
  n_components <- length(fit$components)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_times,
    n_components
  ))
  for (component in 1 : n_components) {
    component_samples[, , component] <- time_varying_mean_samples(
      fit$components[[component]],
      times = times
    )
  }
  if (from == 'categories') {
    output <- .time_varying_mean_mixture_mean_categories(
      component_samples,
      fit$categories
    )
  } else if (from == 'probabilities') {
    output <- .time_varying_mean_mixture_mean_probabilities(
      component_samples,
      component_probabilities(fit)
    )
  }
  attr(output, 'times') <- times
  output
}

#' @export
time_varying_mean_samples.adaptspecmixturefit <- function(
  fit,
  time_step = 1,
  times = seq(1, ntimes(fit), by = time_step),
  from = c('probabilities')
) {
  from <- match.arg(from)
  n_iterations <- nrow(fit$categories)
  n_times <- length(times)
  n_components <- length(fit$components)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_times,
    n_components
  ))
  for (component in 1 : n_components) {
    component_samples[, , component] <- time_varying_mean_samples(
      fit$components[[component]],
      times = times
    )
  }
  if (from == 'probabilities') {
    output <- .time_varying_mean_mixture_samples_probabilities(
      component_samples,
      component_probabilities(fit)
    )
  }
  attr(output, 'times') <- times
  output
}

#' @export
time_varying_spectra_samples.adaptspecmixturefit <- function(
  fit,
  n_frequencies = 64,
  time_step = 1,
  from = c('probabilities', 'categories'),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(fit), by = time_step)
) {
  from <- match.arg(from)
  n_iterations <- nrow(fit$categories)
  n_times <- length(times)
  n_components <- length(fit$components)
  n_frequencies_actual <- length(frequencies)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_frequencies_actual,
    n_times,
    n_components
  ))
  for (component in 1 : n_components) {
    component_samples[, , , component] <- time_varying_spectra_samples(
      fit$components[[component]],
      frequencies = frequencies,
      times = times
    )
  }
  if (from == 'probabilities') {
    output <- .time_varying_spectra_mixture_samples_probabilities(
      component_samples,
      component_probabilities(fit)
    )
  }
  attr(output, 'frequencies') <- frequencies
  attr(output, 'times') <- times
  output
}

#' @export
time_varying_spectra_mean.adaptspecmixturefit <- function(
  fit,
  n_frequencies = 64,
  time_step = 1,
  from = c('probabilities', 'categories'),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(fit), by = time_step)
) {
  from <- match.arg(from)
  n_iterations <- nrow(fit$categories)
  n_times <- length(times)
  n_components <- length(fit$components)
  n_frequencies_actual <- length(frequencies)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_frequencies_actual,
    n_times,
    n_components
  ))
  for (component in 1 : n_components) {
    component_samples[, , , component] <- time_varying_spectra_samples(
      fit$components[[component]],
      frequencies = frequencies,
      times = times
    )
  }
  if (from == 'categories') {
    output <- .time_varying_spectra_mixture_mean_categories(
      component_samples,
      fit$categories
    )
  } else if (from == 'probabilities') {
    output <- .time_varying_spectra_mixture_mean_probabilities(
      component_samples,
      component_probabilities(fit)
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
