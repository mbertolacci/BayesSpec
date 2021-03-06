% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adaptspecmixturefit.R
\name{time_varying_mean_mean.adaptspecmixturefit}
\alias{time_varying_mean_mean.adaptspecmixturefit}
\alias{time_varying_mean_samples.adaptspecmixturefit}
\alias{time_varying_spectra_mean.adaptspecmixturefit}
\alias{time_varying_spectra_samples.adaptspecmixturefit}
\title{Posterior estimates of the time varying mean and spectra from an
adaptspecmixturefit object}
\usage{
\method{time_varying_mean_mean}{adaptspecmixturefit}(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  from = c("probabilities", "categories"),
  ...
)

\method{time_varying_mean_samples}{adaptspecmixturefit}(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  from = c("probabilities"),
  ...
)

\method{time_varying_spectra_mean}{adaptspecmixturefit}(
  x,
  n_frequencies = 64,
  time_step = 1,
  from = c("probabilities", "categories"),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
)

\method{time_varying_spectra_samples}{adaptspecmixturefit}(
  x,
  n_frequencies = 64,
  time_step = 1,
  from = c("probabilities", "categories"),
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
)
}
\arguments{
\item{x}{\code{adaptspecmixturefit} object from \code{\link{adaptspecx}}}

\item{time_step}{Time varying quantity is calculated only at times
divisible by this number. Reduces the size of the output. Ignore if
\code{times} is provided.}

\item{times}{Times at which to calculate the quantity. Must be between
1 and \code{ntimes(x)}, inclusive.}

\item{from}{Whether to calculate the quantity based on the mixture component
weights, or the component indicators for each time series.}

\item{...}{Ignored.}

\item{n_frequencies}{Number of frequencies at which to evaluate the spectral
densities. Ignored if \code{frequencies} is set.}

\item{frequencies}{Frequencies at which to evaluate the spectral density.
Must be between 0 and 0.5, inclusive.}
}
\description{
These methods calculate posterior means or samples of the
time varying mean and spectra from fits performed using AdaptSPEC-X. These
functions can take a lot of time and memory, so consider thinning the input
\code{x} using \code{\link{window.adaptspecfit}} prior to calling, or
using the \code{time_step} argument.
}
\section{Functions}{
\itemize{
\item \code{time_varying_mean_mean.adaptspecmixturefit}: Posterior mean of the

\item \code{time_varying_mean_samples.adaptspecmixturefit}: Posterior samples of
the time varying mean. Returns an array with dimensions (iteration, time,
time series)

\item \code{time_varying_spectra_mean.adaptspecmixturefit}: Posterior mean
estimate of the time varying spectral density. Returns an array with
dimensions (frequency, time, time series).

\item \code{time_varying_spectra_samples.adaptspecmixturefit}: Samples of the time
varying spectral density. Returns a four dimensional with dimensions
(iteration, frequency, time, time series).
}}

\examples{
# Running adaptspecx with i.i.d. normal data
y <- matrix(rnorm(10000), ncol = 10)
design_matrix <- matrix(1, nrow = 10)
fit <- adaptspecx(
  3000,
  1000,
  y,
  design_matrix,
  n_components = 5,
  component_model = adaptspec_model(segment_means = TRUE)
)
fit_thin <- window(fit, thin = 10)
tvmm <- time_varying_mean_mean(fit_thin, time_step = 5)
tvsm <- time_varying_spectra_mean(
  fit_thin,
  n_frequencies = 128,
  time_step = 5
)
par(mfrow = c(1, 2))
matplot(attr(tvmm, 'times'), tvmm, type = 'l')
image(
  t(tvsm[, , 1]),
  x = attr(tvsm, 'times'),
  y = attr(tvsm, 'frequencies'),
  xlab = 'Time',
  ylab = 'Frequency',
  col = terrain.colors(50)
)
}
