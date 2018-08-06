#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, data, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  plotting = FALSE, detrend = TRUE,
  weights_prior = rep(1, n_components),
  start = list(
    weights = NULL,
    categories = NULL,
    components = NULL,
    x_missing = NULL
  ),
  thin = list(
    weights = 1,
    categories = 1,
    n_segments = 1,
    beta = 1,
    tau_squared = 1,
    cut_points = 1,
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  run_diagnostics = TRUE
) {
  thin <- .extend_list(eval(formals(adaptspec_independent_mixture)$thin), thin)

  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices

  ## Prior set up
  # Mixture components
  component_priors <- .mixture_component_priors(component_model, n_components)
  # Validate prior
  .validate_mixture_component_priors(component_priors, n_components, data)
  stopifnot(length(weights_prior) == n_components)

  ## Starting value set up
  start <- .mixture_start(start, component_priors, data, first_category_fixed)
  if (is.null(start$weights)) {
    start$weights <- runif(n_components)
    start$weights[n_components] <- 1 - sum(start$weights[1 : (n_components - 1)])
  }
  # Validate starting values
  .validate_mixture_start(start, n_components, component_priors, data)
  stopifnot(length(start$weights) == n_components)

  # Run sampler
  results <- .independent_mixture(
    n_loop, n_warm_up, data,
    .zero_index_missing_indices(missing_indices),
    component_priors, weights_prior,
    prob_mm1, var_inflate, burn_in_var_inflate,
    first_category_fixed,
    start,
    thin,
    show_progress
  )
  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components

  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors)
  class(results) <- c('adaptspecindependentmixturefit', 'adaptspecmixturefit')

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
window.adaptspecindependentmixturefit <- function(fit, ...) {
  fit <- NextMethod()
  fit$weights <- window(fit$weights, ...)
  fit
}
