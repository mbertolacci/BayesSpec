#' @export
adaptspec_independent_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
  initial_categories = NULL,
  prob_mm1 = 0.8, var_inflate = 1, burn_in_var_inflate = var_inflate,
  first_category_fixed = FALSE,
  n_freq_hat = 50,
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

  results <- .independent_mixture(
    n_loop, n_warm_up, x, component_priors, weight_prior, initial_categories,
    prob_mm1, var_inflate, burn_in_var_inflate,
    first_category_fixed,
    show_progress
  )
  results$n_components <- n_components
  results$weights <- coda::mcmc(aperm(results$weights, c(2, 1)))
  results$categories <- coda::mcmc(aperm(results$categories + 1, c(2, 1)))
  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors, n_freq_hat)

  return(results)
}
