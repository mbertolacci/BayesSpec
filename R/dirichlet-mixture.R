#' @export
adaptspec_dirichlet_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  plotting = FALSE, detrend = TRUE,
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

  if (is.null(initial_categories)) {
    initial_categories <- (0 : (ncol(x) - 1)) %% n_components
  } else if (initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(x), replace = TRUE) - 1
  }
  if (first_category_fixed) {
    # The first time-series is fixed to always be in the first cluster
    initial_categories[1] <- 0
  }

  stopifnot(length(initial_categories) == ncol(x))
  # Cannot allow too many segments
  stopifnot(nrow(x) >= (component_model$n_segments_max * component_model$t_min))

  component_priors <- rep(list(component_model), n_components)
  alpha_prior_shape <- 0.5
  alpha_prior_rate <- 0.5

  missing_indices <- lapply(1 : ncol(x), function(i) which(is.na(x[, i])) - 1)
  results <- .dirichlet_mixture(
    n_loop, n_warm_up, x, missing_indices, component_priors, alpha_prior_shape, alpha_prior_rate,
    initial_categories,
    prob_mm1, var_inflate, burn_in_var_inflate,
    first_category_fixed, thin,
    show_progress
  )
  results$missing_indices <- lapply(missing_indices, function(x) x + 1)
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
