base_component_prior_ind <- list(
  n_segments_max = 10,
  t_min = 40,
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7
)

#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_prior = base_component_prior_ind,
  initial_categories = NULL,
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

  if (is.null(initial_categories)) {
    initial_categories <- (0 : (ncol(data) - 1)) %% n_components
  } else if (initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(data), replace = TRUE) - 1
  }

  # Use default values, override with any provided
  component_prior <- .extend_list(base_component_prior_ind, component_prior)
  component_priors <- rep(list(component_prior), n_components)
  weight_prior <- rep(1, n_components)

  stopifnot(length(initial_categories) == ncol(x))

  results <- .independent_mixture(
    n_loop, n_warm_up, x, component_priors, weight_prior, initial_categories,
    prob_mm1, var_inflate,
    show_progress
  )
  for (component in 1 : n_components) {
    results$components[[component]]$prior <- component_priors[[component]]
    results$components[[component]] <- adaptspecfit(
      results$components[[component]], n_freq_hat
    )
  }
  results$weights <- coda::mcmc(aperm(results$weights, c(2, 1)))
  results$categories <- coda::mcmc(aperm(results$categories, c(2, 1)))

  return(results)
}
