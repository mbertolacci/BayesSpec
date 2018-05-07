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
diagnostic_warnings.adaptspecmixturefit <- function(fit, ...) {
  for (component in 1 : fit$n_components) {
    diagnostic_warnings(
      fit$components[[component]],
      prefix = sprintf('component %d, ', component),
      ...
    )
  }
}
