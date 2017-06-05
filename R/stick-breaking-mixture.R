#' @export
adaptspec_stick_breaking_mixture <- function(
  nloop, nwarmup, nexp_max, x, design_matrix, n_components,
  tmin, sigmasqalpha, tau_prior_a, tau_prior_b, tau_up_limit, prob_mm1,
  step_size_max, var_inflate, nbasis, nfreq_hat,
  initial_categories = NULL,
  plotting = FALSE, detrend = TRUE, nexp_start = 1, show_progress = FALSE
) {
  # For optional variables
  if (missing(sigmasqalpha)) {
    sigmasqalpha <- 100
  }
  if (missing(tau_prior_a)) {
    tau_prior_a <- -1
  }
  if (missing(tau_prior_b)) {
    tau_prior_b <- 0
  }
  if (missing(tau_up_limit)) {
    tau_up_limit <- 10000
  }
  if (missing(prob_mm1)) {
    prob_mm1 <- 0.8
  }
  if (missing(step_size_max)) {
    step_size_max <- 10
  }
  if (missing(var_inflate)) {
    var_inflate <- 1
  }
  if (missing(nbasis)) {
    nbasis <- 7
  }
  if (missing(nfreq_hat)) {
    nfreq_hat <- 50
  }
  if (missing(tmin)) {
    tmin <- 40
  }

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
  }

  priors <- rep(
    list(list(
      n_segments_max = nexp_max,
      t_min = tmin,
      sigma_squared_alpha = sigmasqalpha,
      tau_prior_a = tau_prior_a,
      tau_prior_b = tau_prior_b,
      tau_upper_limit = tau_up_limit,
      n_bases = nbasis
    )),
    n_components
  )
  prior_mean <- c(0)
  prior_precision <- matrix(100, nrow = 1)

  results <- .stick_breaking_mixture(
    nloop, nwarmup, x, design_matrix, priors, prior_mean, prior_precision,
    initial_categories,
    prob_mm1, show_progress
  )
  for (component in 1 : n_components) {
    results$components[[component]]$prior <- priors[[component]]
    results$components[[component]] <- adaptspecfit(
      results$components[[component]], nfreq_hat
    )
  }
  results$beta <- coda::mcmc(aperm(results$beta, c(2, 1)))
  results$alpha <- coda::mcmc(results$alpha)
  results$categories <- coda::mcmc(aperm(results$categories, c(2, 1)))

  return(results)
}
