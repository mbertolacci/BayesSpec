#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, data, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  component_tuning = list(
    prob_short_move = 0.8,
    short_moves = c(-1, 0, 1),
    short_move_weights = c(0.5, 0.5, 0.5),
    var_inflate = 1,
    warm_up_var_inflate = NULL,
    use_cutpoint_within = TRUE,
    use_single_within = FALSE,
    use_hmc_within = FALSE,
    l_min = 1,
    l_max = 10,
    epsilon_min = 0.1,
    epsilon_max = 1,
    use_hessian_curvature = TRUE
  ),
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
    mu = 1,
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

  component_tuning <- .adaptspec_tuning(component_tuning)
  .validate_adaptspec_tuning(component_tuning)

  ## Starting value set up
  if (inherits(start, 'adaptspecindependentmixturefit')) {
    # If provided a chain, continue it
    start <- start$final_values
  } else {
    start <- .mixture_start(
      start,
      component_priors,
      data,
      first_category_fixed,
      component_tuning
    )
    if (is.null(start$weights)) {
      start$weights <- runif(n_components)
      start$weights[n_components] <- (
        1 - sum(start$weights[1 : (n_components - 1)])
      )
    }
  }
  # Validate starting values
  .validate_mixture_start(start, n_components, component_priors, data)
  stopifnot(length(start$weights) == n_components)

  # Run sampler
  results <- .independent_mixture(
    n_loop, n_warm_up, data,
    .zero_index_missing_indices(missing_indices),
    component_priors, weights_prior,
    component_tuning,
    first_category_fixed,
    start,
    thin,
    show_progress
  )
  results$n_time_series <- ncol(data)
  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components
  results$component_tuning <- component_tuning

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

.merge_samples.adaptspecindependentmixturefit <- function(x, fits) {  # nolint
  output <- .merge_samples.adaptspecmixturefit(NULL, fits)  # nolint
  .merge_mcmc_parts(output, fits, c('weights'))
}

#' @export
component_probabilities.adaptspecindependentmixturefit <- function(results) {
  p <- array(0, dim = c(
    nrow(results$weights),
    results$n_time_series,
    ncol(results$weights)
  ))

  for (i in 1 : results$n_time_series) {
    p[, i, ] <- results$weights
  }

  p
}
