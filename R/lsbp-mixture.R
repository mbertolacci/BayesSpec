base_mixture_prior <- list(
  tau_prior_a_squared = 100,
  tau_prior_nu = 3
)

base_spline_prior <- list(
  n_bases = 0
)

#' @export
adaptspec_lsbp_mixture <- function(
  n_loop, n_warm_up, x, design_matrix, n_components,
  spline_group = rep(1, ncol(design_matrix)),
  component_model = adaptspec_model(),
  mixture_prior = base_mixture_prior,
  initial_categories = NULL,
  spline_prior = base_spline_prior,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  n_freq_hat = 50,
  plotting = FALSE, detrend = TRUE, show_progress = FALSE,
  run_diagnostics = TRUE
) {
  x <- as.matrix(x)
  design_matrix <- as.matrix(design_matrix)
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

  # Calculate the spline basis expansion
  spline_prior <- .extend_list(base_spline_prior, spline_prior)
  if (spline_prior$n_bases > 0) {
    # TODO(mgnb): support additive splines; for now, we support just one spline
    stopifnot(all(range(spline_group, na.rm = TRUE) == c(1, 1)))

    non_spline_design_matrix <- design_matrix[, is.na(spline_group), drop = FALSE]
    spline_design_matrix <- design_matrix[, which(spline_group == 1), drop = FALSE]

    if (is.null(spline_prior$type)) {
      spline_prior$type <- ifelse(ncol(spline_design_matrix) == 1, 'smoothing', 'thinplate')
    }
    stopifnot(spline_prior$type %in% c('smoothing', 'thinplate'))

    design_matrix <- cbind(
      1,
      non_spline_design_matrix,
      switch(
        spline_prior$type,
        smoothing = smoothing_spline_basis,
        thinplate = thinplate_spline_basis
      )(spline_design_matrix, spline_prior$n_bases, omit_intercept = TRUE)
    )
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

  component_priors <- rep(list(component_model), n_components)

  mixture_prior <- .extend_list(base_mixture_prior, mixture_prior)
  if (is.null(mixture_prior$mean)) {
    mixture_prior$mean <- matrix(0, nrow = ncol(design_matrix), ncol = n_components - 1)
  }
  if (is.null(mixture_prior$precision)) {
    # For spline fits, these will later be overwritten by estimated of tau
    mixture_prior$precision <- matrix(1 / 100, nrow = ncol(design_matrix), ncol = n_components - 1)
  }

  stopifnot(length(initial_categories) == ncol(x))
  # Cannot allow too many segments
  stopifnot(nrow(x) >= (component_model$n_segments_max * component_model$t_min))

  missing_indices <- lapply(1 : ncol(x), function(i) which(is.na(x[, i])) - 1)
  results <- .lsbp_mixture(
    n_loop, n_warm_up, x, missing_indices, design_matrix, component_priors,
    mixture_prior$mean, mixture_prior$precision,
    mixture_prior$tau_prior_a_squared, mixture_prior$tau_prior_nu,
    initial_categories,
    prob_mm1, var_inflate, burn_in_var_inflate,
    first_category_fixed,
    spline_prior$n_bases, show_progress
  )
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components
  results$design_matrix <- design_matrix
  results$beta <- aperm(results$beta, c(3, 1, 2))
  results$categories <- coda::mcmc(aperm(1 + results$categories, c(2, 1)))
  results$tau_squared <- coda::mcmc(aperm(results$tau_squared, c(2, 1)))
  results$log_posterior <- coda::mcmc(results$log_posterior)

  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors, n_freq_hat)
  class(results) <- c('adaptspeclsbpmixturefit', 'adaptspecmixturefit')

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
window.adaptspeclsbpmixturefit <- function(fit, ...) {
  fit <- NextMethod()
  time_before <- time(fit$tau_squared)
  fit$tau_squared <- window(fit$tau_squared, ...)
  time_after <- time(fit$tau_squared)
  fit$beta <- fit$beta[time_before %in% time_after, , ]
  fit
}

#' @export
component_probabilities.adaptspeclsbpmixturefit <- function(results) {
  beta <- results$beta

  n_components <- dim(beta)[3] + 1
  n_iterations <- dim(beta)[1]

  values <- tensor::tensor(results$design_matrix, beta, 2, 2)
  v <- 1 / (1 + exp(-values))
  p <- array(0, dim = c(nrow(results$design_matrix), n_iterations, n_components))

  v <- 1 / (1 + exp(-values))
  accum <- 1
  for (k in 1 : (n_components - 1)) {
    p[, , k] <- v[, , k] * accum
    accum <- accum * (1 - v[, , k])
  }
  p[, , n_components] <- accum

  # Permute to something sensible
  aperm(p, c(2, 1, 3))
}
