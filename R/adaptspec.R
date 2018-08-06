#' @name adaptspec
#'
#' @title Adaptive Spectral Estimation for Non-stationary Time Series
#'
#' @description Methodology for analyzing possibly non-stationary time series by adaptively dividing the time series into an unknown but finite number of segments and estimating the corresponding local spectra by smoothing splines.
#'
#' @param nloop The total number of MCMC iterations
#' @param nwarmup The number of burn-in iterations
#' @param nexp_max The maximum number of segments allowed
#' @param x The data, a univariate time series, not a time series object
#'
#' @param tmin The minimum number of observations per segment. An optional argument defaulted to tmin = 40.
#' @param sigmasqalpha An optional argument defaulted to sigmasqalpha = 100.
#' @param tau_prior_a An optional argurment defaulted to tau_prior_a = -1.
#' @param tau_prior_b An optional argurment defaulted to tau_prior_b = 0.
#' @param tau_up_limit An optional argurment defaulted to tau_up_limit = 10000.
#' @param prob_mm1 An optional argurment defaulted to prob_mm1 = 0.8.
#' @param step_size_max An optional argurment defaulted to step_size_max = 10.
#' @param var_inflate An optional argurment defaulted to var_inflate = 1.
#' @param nbasis An optional argurment defaulted to nbasis = 7.
#' @param nfreq_hat An optional argurment defaulted to nfreq_hat = 50.
#' @param plotting An optional argument for displaying output plots defaulted to FALSE. When set to TRUE, this displays the spectral and parition points.
#'
#' @return xi The partition points
#' @return log_spec_hat Estimates of the log spectra for all segments
#' @return nexp_curr The number of segments in each iteration.
#'
#' @usage
#' adaptspec(nloop, nwarmup, nexp_max, x,
#'    tmin, sigmasqalpha, tau_prior_a, tau_prior_b,
#'    tau_up_limit, prob_mm1, step_size_max,
#'    var_inflate, nbasis, nfreq_hat, plotting)
#'
#' @examples
#' #Running adaptspec with the simulated_piecewise data.
#' data(simulated_piecewise)
#' model1 <- adaptspec(nloop = 2000, nwarmup = 500,
#'    nexp_max = 10, x = simulated_piecewise)
#' str(model1)
#' summary(model1$nexp_curr)
#' plot(model1$nexp_curr)
#' #Running adaptspec with a sample of the intracranial_eeg data and returing plots and summary.
#' data(intrcranial_eeg)
#' model2 <- adaptspec(nloop = 400, nwarmup = 100,
#'    nexp_max = 20, x = intracranial_eeg[1:2000], plotting = TRUE)
#' summary(model2)
#'
#' @author Rosen, O., Wood, S. and Stoffer, D.
#'
#' @references Rosen, O., Wood, S. and Stoffer, D. (2012). AdaptSPEC: Adaptive Spectral Estimation for Nonstationary Time Series. J. of the American Statistical Association, 107, 1575-1589
#'
#' @export
NULL

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
  # Sampler control
  prob_mm1 = 0.8,
  var_inflate = 1,
  burn_in_var_inflate = var_inflate,
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
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  # Extra
  run_diagnostics = TRUE
) {
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
      time_step = time_step
    ),
    n_loop = n_loop,
    n_warm_up = n_warm_up,
    data = data,
    detrend = detrend,
    prob_mm1 = prob_mm1,
    var_inflate = var_inflate,
    burn_in_var_inflate = burn_in_var_inflate,
    start = start,
    thin = thin,
    show_progress = show_progress,
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
  time_step = 1
) {
  stopifnot(t_min %% time_step == 0)
  model <- list(
    n_segments_min = n_segments_min,
    n_segments_max = n_segments_max,
    t_min = t_min,
    sigma_squared_alpha = sigma_squared_alpha,
    tau_prior_a = tau_prior_a,
    tau_prior_b = tau_prior_b,
    tau_upper_limit = tau_upper_limit,
    n_bases = n_bases,
    time_step = time_step
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
  prob_mm1 = 0.8,
  var_inflate = 1,
  burn_in_var_inflate = var_inflate,
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
    log_posterior = 1,
    x_missing = 1
  ),
  show_progress = FALSE,
  run_diagnostics = TRUE
) {
  thin <- .extend_list(eval(formals(adaptspec_sample)$thin), thin)

  prepared_data <- .prepare_data(data, detrend)
  data <- prepared_data$data
  detrend_fits <- prepared_data$detrend_fits
  missing_indices <- prepared_data$missing_indices

  # Cannot allow too many segments
  stopifnot(nrow(data) >= (model$n_segments_max * model$t_min))

  start <- .adaptspec_start(start, model, data)
  start <- .x_missing_start(start, missing_indices)
  .validate_adaptspec_start(start, model, data)
  .validate_x_missing_start(start, missing_indices)

  results <- .adaptspec(
    n_loop,
    n_warm_up,
    data,
    .zero_index_missing_indices(missing_indices),
    model,
    prob_mm1,
    var_inflate,
    burn_in_var_inflate,
    start,
    thin,
    show_progress
  )

  results$missing_indices <- missing_indices
  results$detrend <- detrend
  results$detrend_fits <- detrend_fits
  results$prob_mm1 <- prob_mm1
  results$var_inflate <- var_inflate
  results$prior <- model
  results <- adaptspecfit(results)

  if (run_diagnostics) diagnostic_warnings(results)

  return(results)
}

#' @export
adaptspec_nu <- function(n_freq, n_bases) {
  splines_basis1d_demmler_reinsch(seq(0, 0.5, length.out = n_freq), n_bases)
}

.adaptspec_start <- function(start, model, data) {
  if (is.null(start$n_segments)) {
    start$n_segments <- sample(model$n_segments_min : model$n_segments_max, 1)
  }
  if (is.null(start$cut_points)) {
    start$cut_points <- rep(nrow(data), model$n_segments_max)
    start$cut_points[1 : start$n_segments] <- model$time_step * floor(
      ((1 : start$n_segments) * nrow(data)) / (start$n_segments * model$time_step)
    )
  }
  if (is.null(start$beta)) {
    start$beta <- matrix(
      rnorm(model$n_segments_max * (1 + model$n_bases)),
      nrow = model$n_segments_max,
      ncol = 1 + model$n_bases
    )
  }
  if (is.null(start$tau_squared)) {
    start$tau_squared <- runif(model$n_segments_max, 0, model$tau_upper_limit)
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
  stopifnot(min(start$tau_squared) > 0)
  stopifnot(max(start$tau_squared) < model$tau_upper_limit)
}
