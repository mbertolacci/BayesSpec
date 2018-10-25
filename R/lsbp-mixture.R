base_mixture_prior <- list(
  tau_prior_a_squared = 100,
  tau_prior_nu = 3
)

base_spline_prior <- list(
  n_bases = 0
)

#' @export
adaptspec_lsbp_mixture <- function(
  n_loop, n_warm_up, data, design_matrix, n_components,
  spline_group = rep(1, ncol(design_matrix)),
  component_model = adaptspec_model(),
  mixture_prior = base_mixture_prior,
  spline_prior = base_spline_prior,
  component_tuning = list(
    prob_short_move = 0.8,
    short_moves = c(-1, 0, 1),
    short_move_weights = c(0.5, 0.5, 0.5),
    var_inflate = 1,
    warm_up_var_inflate = NULL,
    use_cutpoint_within = TRUE,
    use_single_within = FALSE,
    use_hmc_within = FALSE,
    l_min = 190,
    l_max = 210,
    epsilon_min = 0.01,
    epsilon_max = 0.1
  ),
  first_category_fixed = FALSE,
  plotting = FALSE, detrend = TRUE,
  start = list(
    beta = NULL,
    tau_squared = NULL,
    categories = NULL,
    components = NULL,
    x_missing = NULL
  ),
  thin = list(
    beta_lsbp = 1,
    tau_squared_lsbp = 1,
    categories = 1,
    n_segments = 1,
    beta = 1,
    tau_squared = 1,
    cut_points = 1,
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  run_diagnostics = TRUE,
  mpi = FALSE
) {
  thin <- .extend_list(eval(formals(adaptspec_lsbp_mixture)$thin), thin)

  flog.debug('Preparing data', name = 'BayesSpec.lsbp-mixture')
  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices
  design_matrix <- as.matrix(design_matrix)

  ## Prior set up
  # Mixture components
  component_priors <- .mixture_component_priors(component_model, n_components)
  # Calculate the spline basis expansion
  spline_prior <- .extend_list(base_spline_prior, spline_prior)
  if (spline_prior$n_bases > 0) {
    # TODO(mgnb): support additive splines; for now, we support just one spline
    stopifnot(all(range(spline_group, na.rm = TRUE) == c(1, 1)))

    non_spline_design_matrix <- design_matrix[
      ,
      is.na(spline_group),
      drop = FALSE
    ]
    spline_design_matrix <- design_matrix[
      ,
      which(spline_group == 1),
      drop = FALSE
    ]

    if (is.null(spline_prior$type)) {
      spline_prior$type <- ifelse(
        ncol(spline_design_matrix) == 1,
        'smoothing',
        'thinplate'
      )
    }
    stopifnot(spline_prior$type %in% c('smoothing', 'thinplate'))

    flog.debug(
      'Adding spline basis vectors to design matrix',
      name = 'BayesSpec.lsbp-mixture'
    )
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
  mixture_prior <- .extend_list(base_mixture_prior, mixture_prior)
  if (is.null(mixture_prior$mean)) {
    mixture_prior$mean <- matrix(
      0,
      nrow = ncol(design_matrix),
      ncol = n_components - 1
    )
  }
  if (is.null(mixture_prior$precision)) {
    # For spline fits, these will later be overwritten by estimated of tau
    mixture_prior$precision <- matrix(
      1 / 100,
      nrow = ncol(design_matrix),
      ncol = n_components - 1
    )
  }
  # Validate prior
  .validate_mixture_component_priors(component_priors, n_components, data)
  stopifnot(nrow(design_matrix) >= ncol(data))
  stopifnot(nrow(mixture_prior$mean) == ncol(design_matrix))
  stopifnot(ncol(mixture_prior$mean) == n_components - 1)
  stopifnot(nrow(mixture_prior$precision) == ncol(design_matrix))
  stopifnot(ncol(mixture_prior$precision) == n_components - 1)

  ## Starting value set up
  flog.debug('Finding start values', name = 'BayesSpec.lsbp-mixture')
  if (inherits(start, 'adaptspeclsbpmixturefit')) {
    # If provided a chain, continue it
    start <- start$final_values
  } else {
    start <- .mixture_start(
      start,
      component_priors,
      data,
      first_category_fixed,
      initialise_categories = FALSE
    )
    if (is.null(start$categories)) {
      start$categories <- sample.int(
        n_components,
        nrow(design_matrix),
        replace = TRUE
      ) - 1
    }
    if (is.null(start$beta)) {
      start$beta <- matrix(
        rnorm(ncol(design_matrix) * (n_components - 1)),
        nrow = ncol(design_matrix),
        ncol = n_components - 1
      )
    }
    if (is.null(start$tau_squared)) {
      start$tau_squared <- abs(sqrt(mixture_prior$tau_prior_a_squared) * rt(
        n_components - 1,
        mixture_prior$tau_prior_nu
      ))
    }
  }

  if (mpi) {
    futile.logger::flog.debug('Syncing start from rank 0 to all')
    if (Rmpi::mpi.comm.rank(0) == 0) {
      Rmpi::mpi.bcast.Robj(start, 0, 0)
    } else {
      start <- Rmpi::mpi.bcast.Robj(NULL, 0, 0)
    }
  }

  # Validate starting values
  .validate_mixture_start(
    start,
    n_components,
    component_priors,
    data,
    check_categories = FALSE
  )
  stopifnot(length(start$categories) == nrow(design_matrix))
  stopifnot(nrow(start$beta) == ncol(design_matrix))
  stopifnot(ncol(start$beta) == n_components - 1)
  stopifnot(length(start$tau_squared) == n_components - 1)

  component_tuning <- .adaptspec_tuning(component_tuning)
  .validate_adaptspec_tuning(component_tuning)

  flog.debug(
    'Starting MCMC sampler',
    name = 'BayesSpec.lsbp-mixture'
  )
  # Run sampler
  results <- .lsbp_mixture(
    n_loop, n_warm_up, data,
    .zero_index_missing_indices(missing_indices),
    design_matrix, component_priors,
    mixture_prior$mean, mixture_prior$precision,
    mixture_prior$tau_prior_a_squared, mixture_prior$tau_prior_nu,
    component_tuning,
    first_category_fixed,
    spline_prior$n_bases,
    start,
    thin,
    show_progress,
    mpi
  )

  flog.debug('Post-processing MCMC samples', name = 'BayesSpec.lsbp-mixture')
  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$n_components <- n_components
  results$design_matrix <- design_matrix
  results$component_tuning <- component_tuning

  results <- adaptspecmixturefit(results, component_priors)
  class(results) <- c('adaptspeclsbpmixturefit', 'adaptspecmixturefit')

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
window.adaptspeclsbpmixturefit <- function(fit, ...) {
  fit <- NextMethod()
  fit$tau_squared <- window(fit$tau_squared, ...)
  fit$beta <- window(fit$beta, ...)
  fit
}

#' @export
component_probabilities.adaptspeclsbpmixturefit <- function(results) {
  beta <- results$beta

  n_components <- dim(beta)[3] + 1
  n_iterations <- dim(beta)[1]

  values <- tensor::tensor(results$design_matrix, beta, 2, 2)
  v <- 1 / (1 + exp(-values))
  p <- array(0, dim = c(
    nrow(results$design_matrix),
    n_iterations,
    n_components
  ))

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

.merge_samples.adaptspeclsbpmixturefit <- function(x, fits) {  # nolint
  output <- .merge_samples.adaptspecmixturefit(NULL, fits)  # nolint
  .merge_mcmc_parts(output, fits, c('tau_squared', 'beta'))
}
