adaptspecmixturefit <- function(results, component_priors, n_freq_hat) {
  for (component in 1 : results$n_components) {
    results$components[[component]]$prior <- component_priors[[component]]
    results$components[[component]] <- adaptspecfit(
      results$components[[component]], n_freq_hat
    )
  }
  class(results) <- 'adaptspecmixturefit'
  return(results)
}

#' @export
window.adaptspecmixturefit <- function(fit, ...) {
  fit$components <- lapply(fit$components, window, ...)
  fit$categories <- window(fit$categories, ...)
  fit$log_posterior <- window(fit$log_posterior, ...)
  fit
}

#' @export
diagnostics.adaptspecmixturefit <- function(fit, ...) {
  cat(sprintf('Tuning parameters: var_inflate = %f, prob_mm1 = %f\n', fit$var_inflate, fit$prob_mm1))

  cat('Diagnostics for each component:\n')
  for (component in 1 : fit$n_components) {
    cat(sprintf('============ Diagnostics for component %d:\n', component))
    diagnostics(fit$components[[component]], ...)
  }
}

#' @export
diagnostic_plots.adaptspecmixturefit <- function(fit, top = NULL, ...) {
  component_plots <- lapply(1 : fit$n_components, function(component) {
    diagnostic_plots(fit$components[[component]], ...) +
      ggplot2::ggtitle(sprintf('Component %d', component))
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
time_varying_spectra_mean.adaptspecmixturefit <- function(
  fit,
  n_frequencies,
  time_step = 1,
  from = c('categories', 'probabilities')
) {
  from <- match.arg(from)
  n_iterations <- nrow(fit$categories)
  n_time_series <- ncol(fit$categories)
  n_times <- ceiling(max(fit$components[[1]]$cut_points) / time_step)
  n_components <- length(fit$components)

  component_samples <- array(0, dim = c(
    n_iterations,
    n_frequencies,
    n_times,
    n_components
  ))
  for (component in 1 : n_components) {
    component_samples[, , , component] <- time_varying_spectra_samples(
      fit$components[[component]],
      n_frequencies,
      time_step
    )
  }
  if (from == 'categories') {
    output <- .time_varying_spectra_mixture_mean_categories(component_samples, fit$categories)
  } else if (from == 'probabilities') {
    output <- .time_varying_spectra_mixture_mean_probabilities(
      component_samples,
      component_probabilities(fit)
    )
  }
  attr(output, 'frequencies') <- (
    (0 : (n_frequencies - 1)) / (2 * (n_frequencies - 1))
  )
  attr(output, 'times') <- (
    1 + (0 : (n_times - 1)) * time_step
  )
  output
}
