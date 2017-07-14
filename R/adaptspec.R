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
  nloop, nwarmup, nexp_max, x,
  tmin, sigmasqalpha, tau_prior_a, tau_prior_b, tau_up_limit, prob_mm1,
  step_size_max, var_inflate, nbasis, nfreq_hat,
  plotting, detrend = TRUE, nexp_start = 1, show_progress = FALSE
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
  if (missing(plotting)) {
    plotting <- FALSE
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

  prior <- list(
    n_segments_max = nexp_max,
    t_min = tmin,
    sigma_squared_alpha = sigmasqalpha,
    tau_prior_a = tau_prior_a,
    tau_prior_b = tau_prior_b,
    tau_upper_limit = tau_up_limit,
    n_bases = nbasis
  )

  results <- .adaptspec(
    nloop, nwarmup, x, prior, prob_mm1, var_inflate,
    nexp_start, show_progress
  )

  results$prior <- prior
  results <- adaptspecfit(results, nfreq_hat)

  if (plotting) {
    plot(results)
  }

  return(results)
}
