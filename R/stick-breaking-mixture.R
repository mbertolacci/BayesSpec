base_mixture_prior <- list(
  tau_prior_a_squared = 100,
  tau_prior_nu = 3
)

base_spline_prior <- list(
  n_bases = 0
)

#' @export
adaptspec_stick_breaking_mixture <- function(
  n_loop, n_warm_up, x, design_matrix, n_components,
  component_model = adaptspec_model(),
  mixture_prior = base_mixture_prior,
  initial_categories = NULL,
  spline_prior = base_spline_prior,
  prob_mm1 = 0.8, var_inflate = 1, n_freq_hat = 50,
  plotting = FALSE, detrend = TRUE, show_progress = FALSE
) {
  x <- as.matrix(x)
  design_matrix <- as.matrix(design_matrix)

  if (detrend) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    x0 <- 1 : nrow(x)  # nolint
    for (series in 1 : ncol(x)) {
      x[, series] <- lm(x[, series] ~ x0)$res
    }
  }

  # Calculate the spline basis expansion
  spline_prior <- .extend_list(base_spline_prior, spline_prior)
  if (spline_prior$n_bases > 0) {
    if (is.null(spline_prior$type)) {
      spline_prior$type <- ifelse(ncol(design_matrix) == 1, 'smoothing', 'thinplate')
    }
    stopifnot(spline_prior$type %in% c('smoothing', 'thinplate'))

    design_matrix <- switch(
      spline_prior$type,
      smoothing = smoothing_spline_basis,
      thinplate = thinplate_spline_basis
    )(design_matrix, spline_prior$n_bases)
  }

  if (is.null(initial_categories)) {
    initial_categories <- (0 : (ncol(x) - 1)) %% n_components
  } else if (initial_categories == 'random') {
    initial_categories <- sample.int(n_components, ncol(x), replace = TRUE) - 1
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

  results <- .stick_breaking_mixture(
    n_loop, n_warm_up, x, design_matrix, component_priors,
    mixture_prior$mean, mixture_prior$precision,
    mixture_prior$tau_prior_a_squared, mixture_prior$tau_prior_nu,
    initial_categories,
    prob_mm1, var_inflate, spline_prior$n_bases, show_progress
  )
  results$n_components <- n_components
  results$design_matrix <- design_matrix
  results$beta <- aperm(results$beta, c(3, 1, 2))
  results$categories <- coda::mcmc(aperm(1 + results$categories, c(2, 1)))
  results$tau_squared <- coda::mcmc(aperm(results$tau_squared, c(2, 1)))
  results$var_inflate <- var_inflate
  results$prob_mm1 <- prob_mm1

  results <- adaptspecmixturefit(results, component_priors, n_freq_hat)

  return(results)
}

#' @export
component_probabilities <- function(results, thin = 1) {
  beta <- results$beta
  if (thin != 1) {
    beta <- beta[(1 : dim(beta)[1]) %% thin == 0, , ]
  }

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
