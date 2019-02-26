#' @export
adaptspec_dirichlet_mixture <- function(
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
  alpha_prior_shape = 0.5,
  alpha_prior_rate = 0.5,
  start = list(
    log_beta1m = NULL,
    alpha = NULL,
    categories = NULL,
    components = NULL,
    x_missing = NULL
  ),
  thin = list(
    log_beta1m = 1,
    alpha = 1,
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
  thin <- .extend_list(eval(formals(adaptspec_dirichlet_mixture)$thin), thin)

  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices

  ## Prior set up
  # Mixture components
  component_priors <- .mixture_component_priors(component_model, n_components)
  # Validate prior
  .validate_mixture_component_priors(component_priors, n_components, data)

  component_tuning <- .adaptspec_tuning(component_tuning)
  .validate_adaptspec_tuning(component_tuning)

  ## Starting value set up
  if (inherits(start, 'adaptspecdppmixturefit')) {
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
    if (is.null(start$log_beta1m)) {
      start$log_beta1m <- log(runif(n_components))
      start$log_beta1m[n_components] <- -Inf
    }
    if (is.null(start$alpha)) {
      start$alpha <- rgamma(
        1,
        shape = alpha_prior_shape,
        rate = alpha_prior_rate
      )
    }
  }
  # Validate starting values
  .validate_mixture_start(start, n_components, component_priors, data)
  stopifnot(length(start$log_beta1m) == n_components)

  # Run sampler
  results <- .dirichlet_mixture(
    n_loop, n_warm_up, data,
    .zero_index_missing_indices(missing_indices),
    component_priors, alpha_prior_shape, alpha_prior_rate,
    component_tuning,
    first_category_fixed,
    start,
    thin,
    show_progress
  )
  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components
  results$component_tuning <- component_tuning

  results <- adaptspecmixturefit(results, component_priors)
  class(results) <- c('adaptspecdppmixturefit', 'adaptspecmixturefit')

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
window.adaptspecdppmixturefit <- function(fit, ...) {
  fit <- NextMethod()
  fit$log_beta1m <- window(fit$log_beta1m, ...)
  fit$alpha <- window(fit$log_beta1m, ...)
  fit
}

.merge_samples.adaptspecdppmixturefit <- function(x, fits) {  # nolint
  output <- .merge_samples.adaptspecmixturefit(NULL, fits)  # nolint
  .merge_mcmc_parts(output, fits, c('log_beta1m', 'alpha'))
}
