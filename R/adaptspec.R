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
  t_min = 40,
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7,
  # Sampler control
  prob_mm1 = 0.8,
  var_inflate = 1,
  n_freq_hat = 50,
  n_segments_start = max(1, n_segments_min),
  show_progress = FALSE,
  # Extra
  plotting = FALSE
) {
  results <- adaptspec_sample(
    adaptspec_model(
      n_segments_min = n_segments_min,
      n_segments_max = n_segments_max,
      t_min = t_min,
      sigma_squared_alpha = sigma_squared_alpha,
      tau_prior_a = tau_prior_a,
      tau_prior_b = tau_prior_b,
      tau_upper_limit = tau_upper_limit,
      n_bases = n_bases
    ),
    n_loop = n_loop,
    n_warm_up = n_warm_up,
    data = data,
    detrend = detrend,
    prob_mm1 = prob_mm1,
    var_inflate = var_inflate,
    n_freq_hat = n_freq_hat,
    n_segments_start = n_segments_start,
    show_progress = show_progress
  )

  if (plotting) {
    plot(results)
  }

  return(results)
}

#' @export
adaptspec_model <- function(
  n_segments_min = 1,
  n_segments_max = 10,
  t_min = 40,
  sigma_squared_alpha = 100,
  tau_prior_a = -1,
  tau_prior_b = 0,
  tau_upper_limit = 10000,
  n_bases = 7
) {
  model <- list(
    n_segments_min = n_segments_min,
    n_segments_max = n_segments_max,
    t_min = t_min,
    sigma_squared_alpha = sigma_squared_alpha,
    tau_prior_a = tau_prior_a,
    tau_prior_b = tau_prior_b,
    tau_upper_limit = tau_upper_limit,
    n_bases = n_bases
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
  n_freq_hat = 50,
  n_segments_start = model$n_segments_min,
  show_progress = FALSE
) {
  data <- as.matrix(data)
  detrend_fits <- NULL
  if (detrend && ncol(data) > 0) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    data0 <- 1 : nrow(data)  # nolint
    detrend_fits <- list()
    for (series in 1 : ncol(data)) {
      detrend_fits[[series]] <- lm(data[, series] ~ data0)
      data[, series] <- detrend_fits[[series]]$res
    }
  }

  results <- .adaptspec(
    n_loop, n_warm_up, data, model, prob_mm1, var_inflate,
    n_segments_start, show_progress
  )

  results$detrend <- detrend
  results$prob_mm1 <- prob_mm1
  results$var_inflate <- var_inflate
  results$prior <- model
  results$detrend_fits <- detrend_fits
  results <- adaptspecfit(results, n_freq_hat)

  return(results)
}