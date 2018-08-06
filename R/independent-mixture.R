#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, x, n_components,
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

  x <- as.matrix(x)
  detrend_fits <- NULL
  if (detrend && ncol(x) > 0) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    data0 <- 1 : nrow(x)  # nolint
    detrend_fits <- list()
    for (series in 1 : ncol(x)) {
      detrend_fits[[series]] <- lm(x[, series] ~ data0, na.action = na.exclude)
      x[, series] <- residuals(detrend_fits[[series]])
    }
  }

  ## Prior set up
  # Mixture components
  component_priors <- .mixture_component_priors(component_model, n_components)
  # Validate prior
  .validate_mixture_component_priors(component_priors, n_components, x)
  stopifnot(length(weights_prior) == n_components)

  ## Starting value set up
  start <- .mixture_start(start, component_priors, x)
  if (is.null(start$weights)) {
    start$weights <- runif(n_components)
    start$weights[n_components] <- 1 - sum(start$weights[1 : (n_components - 1)])
  }
  # Validate starting values
  .validate_mixture_start(start, n_components, x)
  stopifnot(length(start$weights) == n_components)

  if (first_category_fixed) {
    # The first time-series fixed to always be in the first cluster
    start$categories[1] <- 0
  }

  missing_indices <- .missing_indices(x)
  results <- .independent_mixture(
    n_loop, n_warm_up, x,
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
