#' @export
adaptspec_dirichlet_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
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
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  run_diagnostics = TRUE
) {
  thin <- .extend_list(eval(formals(adaptspec_dirichlet_mixture)$thin), thin)

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

  if (class(component_model) == 'adaptspecmodel') {
    component_priors <- rep(list(component_model), n_components)
  } else {
    component_priors <- component_model
  }
  stopifnot(length(component_priors) == n_components)
  for (component_prior in component_priors) {
    # Cannot allow too many segments
    stopifnot(
      nrow(x) >= (component_model$n_segments_max * component_model$t_min)
    )
  }

  missing_indices <- .missing_indices(x)

  if (is.null(start$log_beta1m)) {
    start$log_beta1m <- log(runif(n_components))
    start$log_beta1m[n_components] <- -Inf
  }
  if (is.null(start$alpha)) {
    start$alpha <- rgamma(1, shape = alpha_prior_shape, rate = alpha_prior_rate)
  }
  if (is.null(start$categories)) {
    start$categories <- sample.int(n_components, ncol(x), replace = TRUE) - 1
  }
  if (is.null(start$components)) {
    start$components <- lapply(component_priors, function(component_prior) {
      .adaptspec_start(NULL, component_prior, x)
    })
  }

  start <- .x_missing_start(start, missing_indices)
  stopifnot(length(start$categories) == ncol(x))
  stopifnot(length(start$components) == n_components)
  stopifnot(length(start$log_beta1m) == n_components)

  if (first_category_fixed) {
    # The first time-series is fixed to always be in the first cluster
    start$categories[1] <- 0
  }

  results <- .dirichlet_mixture(
    n_loop, n_warm_up, x,
    .zero_index_missing_indices(missing_indices),
    component_priors, alpha_prior_shape, alpha_prior_rate,
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
