#' @export
adaptspec_dirichlet_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  n_freq_hat = 50,
  plotting = FALSE, detrend = TRUE, show_progress = FALSE,
  run_diagnostics = TRUE
) {
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
    first_category_fixed,
    show_progress
  )
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components
  results$log_beta1m <- coda::mcmc(aperm(results$log_beta1m, c(2, 1)))
  results$alpha <- coda::mcmc(results$alpha)
  results$categories <- coda::mcmc(aperm(results$categories + 1, c(2, 1)))
  results$log_posterior <- coda::mcmc(results$log_posterior)

  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors, n_freq_hat)
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
