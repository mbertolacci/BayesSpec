#' Adaptive Spectral Estimation for Non-stationary Time Series
#'
#' This function implements a methodology for analyzing possibly non-stationary
#' time series by adaptively dividing the time series into an unknown but finite
#' number of segments and estimating the corresponding local spectra by
#' smoothing splines.
#'
#' @section MCMC tuning parameters:
#' The \code{tuning} argument allows you to change the tuning parameter of the
#' reversible-jump MCMC algorithm used to estimate the model. It is a list with
#' the following entries (and corresponding default values):
#' \itemize{
#'   \item \code{prob_short_move} (\code{0.8}): probability of proposing a small
#'   move of a cutpoint in a within step
#'   \item \code{short_moves} (\code{c(-1, 0, 1)}): set of possible small steps
#'   for the 'small move' proposal. if using time_step > 1, will be multiplied
#'   by time_step.
#'   \item \code{short_move_weights} (\code{c(0.5, 0.5, 0.5)}) the probability
#'   of picking each corresponding entry in \code{short_moves}. Need not be
#'   normalised to sum to one
#'   \item \code{var_inflate} (\code{1}): factor by which to inflate the
#'   covariance matrix of the spline coefficient proposal
#'   \item \code{warm_up_var_inflate} (\code{= var_inflate}): as above, but
#'   applying only during warm-up
#'   \item \code{use_cutpoint_within} (\code{TRUE}): whether to propose the
#'   cutpoints. One would very rarely want to disable this.
#'   \item \code{use_single_within} (\code{FALSE}): whether to also update
#'   spline directly, not just when updating cutpoints
#'   \item \code{use_hmc_within} (\code{FALSE}): whether to also update spline
#'   parameters using HMC
#'   \item \code{l_min} (\code{1}): minimum number of leap frog steps to
#'   take in an HMC iteration. Number of steps is randomly selected from a
#'   discrete uniform over \code{[l_min, l_max]}
#'   \item \code{l_max} (\code{10}): maximum number of lead frog steps to
#'   take in an HMC iteration
#'   \item \code{epsilon_min} (\code{0.1}): minimum step size in HMC. Step size
#'   is chosen from a uniform distribution over
#'   \code{[epsilon_min, epsilon_max]}
#'   \item \code{epsilon_max} (\code{1}): maximum step size in HMC.
#'   \item \code{use_hessian_curvature} (\code{TRUE}): whether to use the
#'   hessian or the Fisher information for curvature of spline fit
#' }
#' If a value is omitted or NULL, the default above will be used.
#'
#' @param n_loop Number of MCMC iterations to perform
#' @param n_warm_up Number of warm-up iterations to discard, so that the
#' number of samples returned is n_loop - n_warm_up
#' @param data Numeric vector or matrix of data. If a matrix, the model assumes
#' it has multiple independent observations of the same random process, one per
#' column.
#' @param detrend Whether to remove a mean and linear trend from the time-series
#' prior to fitting. If \code{data} is a matrix, this will be done independently for
#' each column.
#' @param n_segments_min Minimum number of segments
#' @param n_segments_max Maximum number of segments
#' @param t_min Minimum number of observations per segment
#' @param sigma_squared_alpha Prior variance of smoothing spline intercept
#' @param tau_prior_a Shape of truncated inverse gamma prior for smoothing
#' spline smoothing parameter
#' @param tau_prior_b Scale of truncated inverse gamma prior for smoothing
#' spline smoothing parameter
#' @param tau_upper_limit Upper truncation point for inverse gamma prior for
#' smoothing spline smoothing parameter
#' @param n_bases Number of spline basis vectors to use for log spectrum
#' @param time_step Restricts cut points to times divisible by this number
#' @param frequency_transform How to transform frequencies prior to fitting
#' smoothing spline. Defaults to no transformation.
#' @param segment_means Whether to assume the mean is zero in each segment
#' (FALSE, the default), or estimate segment means
#' @param mu_lower Lower bound of uniform for segment means
#' @param mu_upper Upper bound of uniform for segment means
#' @param tuning Tuning parameters for the MCMC scheme
#' @param start Starting values for MCMC chain. Initialised randomly if blank.
#' Can be provided an adaptspecfit object in order to continue a previous chain.
#' @param thin A list specifying how to thin each output of the MCMC sampler.
#' A value of 1 indicates no thinning, 2 keeps every second sample, and so on.
#' @param show_progress Whether to show a progress indicator during MCMC.
#' @param run_diagnostics Whether to run diagnostics afterwards to determine
#' whether the chain has reached convergence.
#'
#' @return An object of class \code{adaptspecfit}. The MCMC samples are available
#' under the following list entries, each of which are of class
#' \code{\link[coda]{mcmc}} or \code{\link[acoda]{mcmca}}:
#' \itemize{
#'   \item \code{n_segments}: Integer vector with samples of number of segments.
#'   \item \code{cut_points}: Integer matrix with samples of each cut point, with
#'   segment 1 cut point in column 1, and so on. Cells to the right of the
#'   number of segments in that iteration are set to the length of the time
#'   series.
#'   \item \code{mu}: Numeric matrix with samples of segment means. Layout is
#'   as for `cut_points`.
#'   \item \code{tau_squared}: Numeric matrix with samples of smoothing spline
#'   smoothing parameters. Layout is as for `cut_points`.
#'   \item \code{beta}: Three dimensional array with samples of smoothing spline
#'   parameters for each segment.
#'   \item \code{x_missing}: List of numeric matrices, where each matrix holds the
#'   samples of missing values for each column in `data`
#'   \item \code{log_posterior}: Numeric vector of unnormalised values of the log
#'   posterior in each iteration.
#' }
#'
#' @usage
#' adaptspec(
#'   # Sampler
#'   n_loop,
#'   n_warm_up,
#'   # Data
#'   data,
#'   detrend = TRUE,
#'   # Model
#'   n_segments_min = 1,
#'   n_segments_max = 10,
#'   t_min = time_step * floor(40 / time_step),
#'   sigma_squared_alpha = 100,
#'   tau_prior_a = -1,
#'   tau_prior_b = 0,
#'   tau_upper_limit = 10000,
#'   n_bases = 7,
#'   time_step = 1,
#'   frequency_transform = c('identity', 'cbrt'),
#'   segment_means = FALSE,
#'   mu_lower = -1000,
#'   mu_upper = 1000,
#'   # Sampler control
#'   tuning = list(
#'     prob_short_move = 0.8,
#'     short_moves = c(-1, 0, 1),
#'     short_move_weights = c(0.5, 0.5, 0.5),
#'     var_inflate = 1,
#'     warm_up_var_inflate = NULL,
#'     use_cutpoint_within = TRUE,
#'     use_single_within = FALSE,
#'     use_hmc_within = FALSE,
#'     l_min = 1,
#'     l_max = 10,
#'     epsilon_min = 0.1,
#'     epsilon_max = 1,
#'     use_hessian_curvature = TRUE
#'   ),
#'   # Starting values
#'   start = list(
#'     n_segments = NULL,
#'     cut_points = NULL,
#'     beta = NULL,
#'     tau_squared = NULL,
#'     x_missing = NULL
#'   ),
#'   thin = list(
#'     n_segments = 1,
#'     beta = 1,
#'     tau_squared = 1,
#'     cut_points = 1,
#'     log_posterior = 1,
#'     x_missing = 1
#'   ),
#'   show_progress = FALSE,
#'   # Extra
#'   run_diagnostics = TRUE
#' )
#' @examples
#' # Running adaptspec with the simulated_piecewise data.
#' data(simulated_piecewise)
#' model1 <- adaptspec(
#'   5000,
#'   1000,
#'   simulated_piecewise
#' )
#' summary(model1)
#' @seealso \code{\link{segment_log_spectra_mean}} for methods calculating
#' estimates of the spectral densities in each segments.
#' \code{\link{time_varying_spectra_mean}} for estimates of the time-varying
#' spectral density. \code{\link{adaptspecfit}} for other useful methods
#' applying to objects returned by this function.
#' @export
adaptspec <- function(
  # Sampler
  n_loop,
  n_warm_up,
  # Data
  data,
  detrend = TRUE,
  # Model
  n_segments_min = 1,
  n_segments_max = 10,
  t_min = time_step * floor(40 / time_step),
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7,
  time_step = 1,
  frequency_transform = c('identity', 'cbrt'),
  segment_means = FALSE,
  mu_lower = -1000,
  mu_upper = 1000,
  # Sampler control
  tuning = list(
    prob_short_move = 0.8,
    short_moves = c(-1, 0, 1),
    short_move_weights = c(0.5, 0.5, 0.5),
    var_inflate = 1,
    warm_up_var_inflate = NULL,
    use_single_within = FALSE,
    use_hmc_within = FALSE,
    l_min = 1,
    l_max = 10,
    epsilon_min = 0.1,
    epsilon_max = 1,
    use_hessian_curvature = TRUE
  ),
  # Starting values
  start = list(
    n_segments = NULL,
    cut_points = NULL,
    beta = NULL,
    tau_squared = NULL,
    x_missing = NULL
  ),
  thin = list(
    n_segments = 1,
    beta = 1,
    tau_squared = 1,
    cut_points = 1,
    mu = 1,
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  debug = FALSE,
  # Extra
  run_diagnostics = TRUE
) {
  frequency_transform <- match.arg(frequency_transform)
  adaptspec_sample(
    adaptspec_model(
      n_segments_min = n_segments_min,
      n_segments_max = n_segments_max,
      t_min = t_min,
      sigma_squared_alpha = sigma_squared_alpha,
      tau_prior_a = tau_prior_a,
      tau_prior_b = tau_prior_b,
      tau_upper_limit = tau_upper_limit,
      n_bases = n_bases,
      time_step = time_step,
      frequency_transform = frequency_transform,
      segment_means = segment_means,
      mu_lower = mu_lower,
      mu_upper = mu_upper
    ),
    n_loop = n_loop,
    n_warm_up = n_warm_up,
    data = data,
    detrend = detrend,
    tuning = tuning,
    start = start,
    thin = thin,
    show_progress = show_progress,
    debug = debug,
    run_diagnostics = run_diagnostics
  )
}

#' @export
adaptspec_model <- function(
  n_segments_min = 1,
  n_segments_max = 10,
  t_min = time_step * floor(40 / time_step),
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7,
  time_step = 1,
  frequency_transform = c('identity', 'cbrt'),
  segment_means = FALSE,
  mu_lower = -1000,
  mu_upper = 1000
) {
  stopifnot(t_min %% time_step == 0)
  frequency_transform <- match.arg(frequency_transform)
  model <- list(
    n_segments_min = n_segments_min,
    n_segments_max = n_segments_max,
    t_min = t_min,
    sigma_squared_alpha = sigma_squared_alpha,
    tau_prior_a = tau_prior_a,
    tau_prior_b = tau_prior_b,
    tau_upper_limit = tau_upper_limit,
    n_bases = n_bases,
    time_step = time_step,
    frequency_transform = frequency_transform,
    segment_means = segment_means,
    mu_lower = mu_lower,
    mu_upper = mu_upper
  )
  class(model) <- 'adaptspecmodel'
  return(model)
}

#' @export
adaptspec_sample <- function(
  model,
  n_loop,
  n_warm_up,
  data,
  detrend = TRUE,
  tuning = list(
    prob_mm1 = 0.8,
    var_inflate = 1,
    warm_up_var_inflate = NULL
  ),
  start = list(
    n_segments = NULL,
    cut_points = NULL,
    beta = NULL,
    tau_squared = NULL,
    x_missing = NULL
  ),
  thin = list(
    n_segments = 1,
    beta = 1,
    tau_squared = 1,
    cut_points = 1,
    mu = 1,
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  debug = FALSE,
  run_diagnostics = TRUE
) {
  thin <- .extend_list(eval(formals(adaptspec_sample)$thin), thin)

  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices

  # Cannot allow too many segments
  stopifnot(nrow(data) >= (model$n_segments_max * model$t_min))

  tuning <- .adaptspec_tuning(tuning)
  .validate_adaptspec_tuning(tuning)

  if (inherits(start, 'adaptspecfit')) {
    # If provided a chain, continue it
    start <- start$final_values
  } else {
    start <- .adaptspec_start(start, model, data, tuning)
    start <- .x_missing_start(start, missing_indices)
  }
  .validate_adaptspec_start(start, model, data)
  .validate_x_missing_start(start, missing_indices)

  results <- .adaptspec(
    n_loop,
    n_warm_up,
    data,
    .zero_index_missing_indices(missing_indices),
    model,
    tuning,
    start,
    thin,
    show_progress,
    debug
  )

  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$tuning <- tuning
  results$prior <- model
  results <- adaptspecfit(results)

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
adaptspec_nu <- function(n_freq, n_bases) {
  splines_basis1d_demmler_reinsch(2 * seq(0, 0.5, length.out = n_freq), n_bases)
}

.adaptspec_start <- function(start, model, data, tuning) {
  if (is.null(start$n_segments)) {
    if (model$n_segments_min == model$n_segments_max) {
      start$n_segments <- model$n_segments_min
    } else {
      start$n_segments <- sample(model$n_segments_min : model$n_segments_max, 1)
    }
  }
  if (is.null(start$cut_points)) {
    start$cut_points <- rep(nrow(data), model$n_segments_max)
    start$cut_points[1 : start$n_segments] <- model$time_step * floor(
      ((1 : start$n_segments) * nrow(data)) /
      (start$n_segments * model$time_step)
    )
  }
  if (is.null(start$tau_squared)) {
    start$tau_squared <- rep(0, model$n_segments_max)
    start$tau_squared[1 : start$n_segments] <- runif(
      length(1 : start$n_segments),
      0,
      model$tau_upper_limit
    )
  }
  if (is.null(start$mu)) {
    start$mu <- rep(0, model$n_segments_max)
    if (model$segment_means) {
      start$mu[1 : start$n_segments] <- sapply(
        1 : start$n_segments,
        function(segment) {
          mean(data[
            (
              if (segment == 1) 1 else (start$cut_points[segment - 1] + 1)
            ) : start$cut_points[segment],
          ], na.rm = TRUE)
        }
      )
      start$mu[!is.finite(start$mu)] <- 0
    }
  }
  if (is.null(start$beta)) {
    start$beta <- matrix(
      0,
      nrow = model$n_segments_max,
      ncol = 1 + model$n_bases
    )
    for (n_segments in 1 : start$n_segments) {
      start$beta[n_segments, ] <- rnorm(1 + model$n_bases)
    }
    # Runs the optimisation algorithm to find the conditional mode for beta
    data[is.na(data)] <- rnorm(sum(is.na(data)))
    state <- .get_sample_filled(data, model, start, tuning)
    start$beta <- state$beta_mode
  }

  start
}

.validate_adaptspec_start <- function(start, model, data) {
  stopifnot(length(start$n_segments) == 1)
  stopifnot(start$n_segments >= model$n_segments_min)
  stopifnot(start$n_segments <= model$n_segments_max)

  stopifnot(length(start$cut_points) == model$n_segments_max)
  stopifnot(min(start$cut_points) >= model$t_min)
  stopifnot(max(start$cut_points) <= nrow(data))
  stopifnot(all(order(start$cut_points) == (1 : model$n_segments_max)))

  stopifnot(is.matrix(start$beta))
  stopifnot(nrow(start$beta) == model$n_segments_max)
  stopifnot(ncol(start$beta) == 1 + model$n_bases)
  stopifnot(!anyNA(start$beta))

  stopifnot(is.numeric(start$tau_squared))
  stopifnot(length(start$tau_squared) == model$n_segments_max)
  stopifnot(min(start$tau_squared) >= 0)
  stopifnot(min(start$tau_squared[model$n_segments_min : start$n_segments]) > 0)
  stopifnot(max(start$tau_squared) < model$tau_upper_limit)
}

.adaptspec_tuning <- function(tuning) {
  tuning <- .extend_list(list(
    prob_short_move = 0.8,
    short_moves = c(-1, 0, 1),
    short_move_weights = NULL,
    var_inflate = 1,
    use_cutpoint_within = TRUE,
    use_single_within = FALSE,
    use_hmc_within = FALSE,
    l_min = 1,
    l_max = 10,
    epsilon_min = 0.1,
    epsilon_max = 1,
    use_hessian_curvature = TRUE
  ), tuning)

  tuning$short_moves <- as.integer(tuning$short_moves)
  if (is.null(tuning$short_move_weights)) {
    tuning$short_move_weights <- rep(1, length(tuning$short_moves))
  }

  if (is.null(tuning$warm_up_var_inflate)) {
    tuning$warm_up_var_inflate <- tuning$var_inflate
  }
  tuning$l_min <- as.integer(tuning$l_min)
  tuning$l_max <- as.integer(tuning$l_max)
  tuning
}

.validate_adaptspec_tuning <- function(tuning) {
  # NOTE(mgnb): lintr can't tell that these variables exist within the tuning
  # object
  # nolint start
  with(tuning, {
    stopifnot(prob_short_move >= 0 && prob_short_move <= 1)
    stopifnot(is.integer(short_moves))
    stopifnot(!anyNA(short_moves))
    stopifnot(is.numeric(short_move_weights))
    stopifnot(!anyNA(short_move_weights))
    stopifnot(length(short_moves) == length(short_move_weights))

    stopifnot(is.numeric(var_inflate))
    stopifnot(!is.na(var_inflate))
    stopifnot(is.numeric(warm_up_var_inflate))
    stopifnot(!is.na(warm_up_var_inflate))

    stopifnot(is.logical(use_single_within))

    stopifnot(is.logical(use_hmc_within))
    stopifnot(is.integer(l_min))
    stopifnot(is.integer(l_max))
    stopifnot(l_min <= l_max)
    stopifnot(is.numeric(epsilon_min))
    stopifnot(is.numeric(epsilon_max))
    stopifnot(epsilon_min <= epsilon_max)

    stopifnot(is.logical(use_hessian_curvature))
  })
  # nolint end
}
