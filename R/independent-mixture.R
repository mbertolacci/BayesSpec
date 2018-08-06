#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  plotting = FALSE, detrend = TRUE,
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

  if (is.null(initial_categories)) {
    initial_categories <- (0 : (ncol(x) - 1)) %% n_components
  } else if (is.character(initial_categories) && initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(x), replace = TRUE) - 1
  }
  if (first_category_fixed) {
    # The first time-series is fixed to always be in the first cluster
    initial_categories[1] <- 0
  }

  component_priors <- rep(list(component_model), n_components)
  weight_prior <- rep(1, n_components)

  stopifnot(length(initial_categories) == ncol(x))
  # Cannot allow too many segments
  stopifnot(nrow(x) >= (component_model$n_segments_max * component_model$t_min))

  missing_indices <- .missing_indices(x)
  results <- .independent_mixture(
    n_loop, n_warm_up, x,
    .zero_index_missing_indices(missing_indices),
    component_priors, weight_prior, initial_categories,
    prob_mm1, var_inflate, burn_in_var_inflate,
    first_category_fixed,
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
