base_component_prior <- list(
  n_segments_max = 10,
  t_min = 40,
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7
)

base_mixture_prior <- list(
  tau_prior_a_squared = 100,
  tau_prior_nu = 3
)

#' @export
adaptspec_stick_breaking_mixture <- function(
  n_loop, n_warm_up, x, design_matrix, n_components,
  component_prior = base_component_prior,
  mixture_prior = base_mixture_prior,
  initial_categories = NULL,
  n_spline_bases = 0,
  prob_mm1 = 0.8, var_inflate = 1, n_freq_hat = 50,
  plotting = FALSE, detrend = TRUE, show_progress = FALSE
) {
  x <- as.matrix(x)
  if (detrend) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    x0 <- 1 : nrow(x)  # nolint
    for (series in 1 : ncol(x)) {
      x[, series] <- lm(x[, series] ~ x0)$res
    }
  }

  if (n_spline_bases > 0) {
    stopifnot(nrow(design_matrix) >= n_spline_bases)
    stopifnot(ncol(design_matrix) <= 2)

    if (ncol(design_matrix) == 1) {
      design_matrix <- splines_basis1d(design_matrix, n_spline_bases)
    } else {
      design_matrix <- splines_thinplate(design_matrix, n_spline_bases)$design_matrix
    }
  }

  if (is.null(initial_categories)) {
    initial_categories <- (0 : (ncol(data) - 1)) %% n_components
  } else if (initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(data), replace = TRUE) - 1
  }

  # Use default values, override with any provided
  component_prior <- .extend_list(base_component_prior, component_prior)
  component_priors <- rep(list(component_prior), n_components)

  mixture_prior <- .extend_list(base_mixture_prior, mixture_prior)
  if (is.null(mixture_prior$mean)) {
    mixture_prior$mean <- matrix(0, nrow = ncol(design_matrix), ncol = n_components - 1)
  }
  if (is.null(mixture_prior$precision)) {
    # For spline fits, these will later be overwritten by estimated of tau
    mixture_prior$precision <- matrix(1 / 100, nrow = ncol(design_matrix), ncol = n_components - 1)
  }

  stopifnot(length(initial_categories) == ncol(x))

  results <- .stick_breaking_mixture(
    n_loop, n_warm_up, x, design_matrix, component_priors,
    mixture_prior$mean, mixture_prior$precision,
    mixture_prior$tau_prior_a_squared, mixture_prior$tau_prior_nu,
    initial_categories,
    prob_mm1, var_inflate, n_spline_bases, show_progress
  )
  for (component in 1 : n_components) {
    results$components[[component]]$prior <- component_priors[[component]]
    results$components[[component]] <- adaptspecfit(
      results$components[[component]], n_freq_hat
    )
  }
  results$design_matrix <- design_matrix
  results$beta <- aperm(results$beta, c(3, 1, 2))
  results$categories <- coda::mcmc(aperm(results$categories, c(2, 1)))
  results$tau_squared <- coda::mcmc(aperm(results$tau_squared, c(2, 1)))

  return(results)
}
