#' @export
adaptspec_dirichlet_mixture <- function(
  n_loop, n_warm_up, x, n_components,
  component_model = adaptspec_model(),
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
    initial_categories <- (0 : (ncol(x) - 1)) %% n_components
  } else if (initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(x), replace = TRUE) - 1
  }
  stopifnot(length(initial_categories) == ncol(x))

  component_priors <- rep(list(component_model), n_components)
  alpha_prior_shape <- 0.5
  alpha_prior_rate <- 0.5


  results <- .dirichlet_mixture(
    n_loop, n_warm_up, x, component_priors, alpha_prior_shape, alpha_prior_rate,
    initial_categories,
    prob_mm1, var_inflate,
    show_progress
  )
  results$n_components <- n_components
  results$beta <- coda::mcmc(aperm(results$beta, c(2, 1)))
  results$alpha <- coda::mcmc(results$alpha)
  results$categories <- coda::mcmc(aperm(results$categories + 1, c(2, 1)))

  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors, n_freq_hat)

  return(results)
}
