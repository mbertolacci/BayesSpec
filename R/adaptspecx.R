#' AdaptSPEC-X: Covariate Dependent Spectral Modeling of Multiple Nonstationary Time Series
#'
#' This function implements methodology for analyzing multiple possibly
#' non-stationary time series that are indexed by covariates using a covariate
#' dependent infinite mixture. The mixture components are based on AdaptSPEC
#' It is described by \href{https://arxiv.org/abs/1908.06622}{Bertolacci et al., 2020}.
#'
#' @param n_loop Number of MCMC iterations to perform
#' @param n_warm_up Number of warm-up iterations to discard, so that the
#' number of samples returned is n_loop - n_warm_up
#' @param data Numeric vector or matrix of data. If a matrix, the model assumes
#' it has multiple independent observations of the same random process, one per
#' column.
#' @param design_matrix Design matrix of covariates. There should be one row per
#' column of \code{data}.
#' @param n_components The number of mixture components at which to truncate the
#' infinite mixture.
#' @param detrend Whether to remove a mean and linear trend from the time-series
#' prior to fitting. If \code{data} is a matrix, this will be done independently for
#' each column.
#' @param spline_covariates A logical vector of length \code{ncol(design_matrix)}
#' indicating which covariates to include in the spline (if any).
#' @param component_model The AdaptSPEC model to be used for each mixture
#' component. See \code{\link{adaptspec_model}}.
#' @param mixture_prior The prior on the mixture component parameters.
#' @param component_tuning Tuning parameters for the AdaptSPEC component MCMC
#' steps; see \code{\link{adaptspec_tuning}}
#' @param lsbp_tuning Tuning parameters for the LSBP sampler. You can set the
#' number of swap moves (\code{n_swap_moves}) to attempt on each iteration, and
#' how far to consider swapping components (\code{swap_move_length}).
#' @param start List of starting values; either an object returned from this
#' function, or a list. Values that are NULL/missing are ignored.
#' @param thin A list specifying how to thin each output of the MCMC sampler.
#' A value of 1 indicates no thinning, 2 keeps every second sample, and so on.
#' @param show_progress Whether to show a progress indicator during MCMC
#' @param run_diagnostics Whether to run diagnostics afterwards to help
#' determine whether the chain has reached convergence.
#'
#' @return An object of class \code{adaptspeclsbpmixturefit}. The MCMC samples
#' are available under the following list entries, most of which are of class
#' \code{\link[coda]{mcmc}} or \code{\link[acoda]{mcmca}}:
#' \itemize{
#'   \item \code{beta}: Three dimensional array with samples of coefficients
#'   for the covariate dependent logistic stick breaking prior, including
#'   spline coefficients if used.
#'   \item \code{tau_squared}: Matrix of samples of scale parameter for splin
#'   coefficients, if used.
#'   \item \code{components}: List of objects of class
#'   \code{\link[=adaptspec]{adaptspecfit}} containing the samples for
#'   each mixture component
#'   \item \code{x_missing}: List of numeric matrices, where each matrix holds
#'   the samples of missing values for each column in `data`
#'   \item \code{log_posterior}: Numeric vector of unnormalised values of the
#'   log posterior in each iteration.
#' }
#'
#' @examples
#' # Running adaptspecx with i.i.d. normal data
#' y <- matrix(rnorm(10000), ncol = 10)
#' design_matrix <- matrix(1, nrow = 10)
#' fit <- adaptspecx(
#'   5000,
#'   1000,
#'   y,
#'   design_matrix,
#'   n_components = 5
#' )
#' @seealso \code{\link{segment_log_spectra_mean}} for methods calculating
#' estimates of the spectral densities in each segments.
#' \code{\link{time_varying_spectra_mean.adaptspecmixturefit}} for estimates of
#' the time-varying spectral density. \code{\link{adaptspecmixturefit}} for
#' other useful methods applying to objects returned by this function.
#' \code{\link{merge_samples}} to merge samples from independent runs of this
#' function.
#' \code{\link{component_probabilities}} to calculate the probabilities of
#' component membership.
#' @export
adaptspecx <- function(
  # Sampler
  n_loop,
  n_warm_up,
  # Data
  data,
  design_matrix,
  # Model
  n_components,
  detrend = FALSE,
  spline_covariates = rep(TRUE, ncol(design_matrix)),
  component_model = adaptspec_model(),
  mixture_prior = list(
    mean = 0,
    precision = 1 / 100,
    tau_prior_a_squared = 100,
    tau_prior_nu = 3,
    tau_prior_upper = 10000,
    n_spline_bases = 0,
    spline_type = 'thinplate'
  ),
  # Sampler control
  component_tuning = adaptspec_tuning(),
  lsbp_tuning = list(
    n_swap_moves = 1,
    swap_move_length = 3
  ),
  # Starting values
  start = list(
    beta = NULL,
    tau_squared = NULL,
    categories = NULL,
    components = NULL,
    x_missing = NULL
  ),
  # Extra
  thin = list(
    beta_lsbp = 1,
    tau_squared_lsbp = 1,
    categories = 1,
    n_segments = 1,
    beta = 1,
    tau_squared = 1,
    cut_points = 1,
    mu = 1,
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  run_diagnostics = TRUE
) {
  thin <- .extend_list(eval(formals(adaptspecx)$thin), thin)
  lsbp_tuning <- .extend_list(
    eval(formals(adaptspecx)$lsbp_tuning),
    lsbp_tuning
  )

  # Currently unexported functionality
  lsbp_tuning$n_split_merge_moves <- 0L
  first_category_fixed <- FALSE
  mpi <- FALSE

  flog.debug('Preparing data', name = 'BayesSpec.lsbp-mixture')
  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices
  design_matrix <- as.matrix(design_matrix)

  n_time_series <- ncol(data)

  ## Prior set up
  # Mixture components
  component_priors <- .mixture_component_priors(component_model, n_components)
  mixture_prior <- .extend_list(
    eval(formals(adaptspecx)$mixture_prior),
    mixture_prior
  )

  # Calculate the spline basis expansion
  if (mixture_prior$n_spline_bases > 0) {
    non_spline_design_matrix <- design_matrix[
      ,
      !spline_covariates,
      drop = FALSE
    ]
    spline_design_matrix <- design_matrix[
      ,
      spline_covariates,
      drop = FALSE
    ]

    if (is.null(mixture_prior$spline_type)) {
      mixture_prior$spline_type <- ifelse(
        ncol(spline_design_matrix) == 1,
        'smoothing',
        'thinplate'
      )
    }
    stopifnot(mixture_prior$spline_type %in% c('smoothing', 'thinplate'))

    flog.debug(
      'Adding spline basis vectors to design matrix',
      name = 'BayesSpec.lsbp-mixture'
    )
    design_matrix <- cbind(
      1,
      non_spline_design_matrix,
      switch(
        mixture_prior$spline_type,
        smoothing = .smoothing_spline_basis,
        thinplate = .thinplate_spline_basis
      )(spline_design_matrix, mixture_prior$n_spline_bases, omit_intercept = TRUE)
    )
  }

  mixture_prior$mean <- matrix(
    mixture_prior$mean,
    nrow = ncol(design_matrix),
    ncol = n_components - 1
  )
  # For spline fits, these will later be overwritten by estimated of tau
  mixture_prior$precision <- matrix(
    mixture_prior$precision,
    nrow = ncol(design_matrix),
    ncol = n_components - 1
  )
  # Validate prior
  .validate_mixture_component_priors(component_priors, n_components, data)
  stopifnot(nrow(design_matrix) >= ncol(data))
  stopifnot(nrow(mixture_prior$mean) == ncol(design_matrix))
  stopifnot(ncol(mixture_prior$mean) == n_components - 1)
  stopifnot(nrow(mixture_prior$precision) == ncol(design_matrix))
  stopifnot(ncol(mixture_prior$precision) == n_components - 1)

  .validate_adaptspec_tuning(component_tuning)

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
      component_tuning,
      initialise_categories = FALSE
    )
    if (mixture_prior$n_spline_bases > 0 && is.null(start$tau_squared)) {
      while (TRUE) {
        start$tau_squared <- abs(sqrt(mixture_prior$tau_prior_a_squared) * stats::rt(
          n_components - 1,
          mixture_prior$tau_prior_nu
        ))
        if (all(start$tau_squared <= mixture_prior$tau_prior_upper)) {
          break
        }
      }
      spline_indices <- (
        nrow(mixture_prior$precision) - mixture_prior$n_spline_bases + 1
      ) : nrow(mixture_prior$precision)
      # Update the mixture prior with the chosen precisions
      for (k in seq_along(start$tau_squared)) {
        mixture_prior$precision[spline_indices, k] <- 1 / start$tau_squared[k]
      }
    }
    if (is.null(start$beta)) {
      start$beta <- matrix(
        NA,
        nrow = ncol(design_matrix),
        ncol = n_components - 1
      )
      for (k in seq_len(ncol(start$beta))) {
        start$beta[, k] <- stats::rnorm(
          ncol(design_matrix),
          sd = 1 / sqrt(mixture_prior$precision[, k])
        )
      }
    }
    if (is.null(start$categories)) {
      start$categories <- sample.int(
        n_components,
        n_time_series,
        replace = TRUE
      ) - 1
    }
    if (first_category_fixed) {
      start$categories[1] <- 0
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
  stopifnot(length(start$categories) == n_time_series)
  stopifnot(nrow(start$beta) == ncol(design_matrix))
  stopifnot(ncol(start$beta) == n_components - 1)
  if (mixture_prior$n_spline_bases > 0) {
    stopifnot(length(start$tau_squared) == n_components - 1)
  }

  flog.debug(
    'Starting MCMC sampler',
    name = 'BayesSpec.lsbp-mixture'
  )
  # Run sampler
  results <- .lsbp_mixture(
    n_loop, n_warm_up, data,
    .zero_index_missing_indices(missing_indices),
    design_matrix[1 : n_time_series, , drop = FALSE],
    component_priors,
    mixture_prior$mean, mixture_prior$precision,
    mixture_prior$tau_prior_a_squared, mixture_prior$tau_prior_nu,
    mixture_prior$tau_prior_upper,
    component_tuning,
    lsbp_tuning,
    first_category_fixed,
    mixture_prior$n_spline_bases,
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
  results$mixture_prior <- mixture_prior

  results <- .adaptspecmixturefit(results, component_priors)
  class(results) <- c('adaptspeclsbpmixturefit', 'adaptspecmixturefit')

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
window.adaptspeclsbpmixturefit <- function(x, ...) {
  x <- NextMethod()
  x$tau_squared <- window(x$tau_squared, ...)
  x$beta <- window(x$beta, ...)
  x
}

#' Calculate mixture probabilities for each time series for AdaptSPEC-X
#'
#' This function calculates samples from the probability of component
#' membership for each time series for a fit performed by
#' \code{\link{adaptspec}}.
#'
#' @param x The object to calculate the probabilities.
#' @return An array with dimensions (iteration, time series, component)
#' @export
component_probabilities <- function(x) {
  beta <- x$beta

  n_components <- dim(beta)[3] + 1
  n_iterations <- dim(beta)[1]

  values <- tensor::tensor(x$design_matrix, beta, 2, 2)
  v <- 1 / (1 + exp(-values))
  p <- array(0, dim = c(
    nrow(x$design_matrix),
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

#' @export
diagnostic_plots.adaptspeclsbpmixturefit <- function(x, ...) {
  component_plots <- diagnostic_plots.adaptspecmixturefit(
    x,
    top = 'Spectra splines',
    ...
  )
  tau_squared_df <- do.call(rbind, lapply(
    seq_len(ncol(x$tau_squared)),
    function(i) {
      data.frame(
        iteration = as.vector(time(x$tau_squared)),
        component = i,
        value = as.vector(x$tau_squared[, i]),
        stringsAsFactors = FALSE
      )
    }
  ))
  tau_squared_plot <- ggplot2::ggplot(
    tau_squared_df,
    ggplot2::aes(iteration, value)
  ) +
    ggplot2::geom_line() +
    ggplot2::facet_wrap(
      ~ component,
      scales = 'free_y',
      labeller = ggplot2::label_both,
      ncol = 1
    ) +
    ggplot2::ggtitle('Tau squared')

  get_beta_df <- function(i) {
    do.call(rbind, lapply(
      seq_len(dim(x$beta)[3]),
      function(j) {
        data.frame(
          iteration = as.vector(stats::time(x$beta)),
          component = j,
          column = i,
          value = as.vector(x$beta[, i, j]),
          stringsAsFactors = FALSE
        )
      }
    ))
  }
  beta_df <- do.call(rbind, lapply(1 : 5, get_beta_df))
  beta_plot <- ggplot2::ggplot(
    beta_df,
    ggplot2::aes(iteration, value)
  ) +
    ggplot2::geom_line() +
    ggplot2::facet_wrap(
      ~ component + column,
      scales = 'free',
      labeller = ggplot2::label_both,
      ncol = 5
    ) +
    ggplot2::ggtitle('Beta')

  gridExtra::grid.arrange(
    component_plots,
    tau_squared_plot,
    beta_plot,
    widths = c(12, 1, 4),
    ncol = 3
  )
}

.merge_samples.adaptspeclsbpmixturefit <- function(x, fits) {  # nolint
  output <- .merge_samples.adaptspecmixturefit(NULL, fits)  # nolint
  .merge_mcmc_parts(output, fits, c('tau_squared', 'beta'))
}
