% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adaptspecfit.R
\name{time_varying_mean_mean.adaptspecfit}
\alias{time_varying_mean_mean.adaptspecfit}
\alias{time_varying_mean_samples.adaptspecfit}
\alias{time_varying_spectra_mean.adaptspecfit}
\alias{time_varying_spectra_samples.adaptspecfit}
\title{Posterior estimates of the time varying mean and spectra from an
adaptspecfit object}
\usage{
\method{time_varying_mean_mean}{adaptspecfit}(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  ...
)

\method{time_varying_mean_samples}{adaptspecfit}(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  ...
)

\method{time_varying_spectra_mean}{adaptspecfit}(
  x,
  n_frequencies = 64,
  time_step = 1,
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
)

\method{time_varying_spectra_samples}{adaptspecfit}(
  x,
  n_frequencies = 64,
  time_step = 1,
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
)
}
\arguments{
\item{x}{\code{adaptspecfit} object}

\item{time_step}{Time varying quantity is calculated only at times
divisible by this number. Reduces the size of the output. Ignore if
\code{times} is provided.}

\item{times}{Times at which to calculate the quantity. Must be between
1 and \code{ntimes(x)}, inclusive.}

\item{...}{Ignored.}

\item{n_frequencies}{Number of frequencies at which to evaluate the spectral
densities. Ignored if \code{frequencies} is set.}

\item{frequencies}{Frequencies at which to evaluate the spectral density.
Must be between 0 and 0.5, inclusive.}
}
\description{
These methods calculate posterior means or samples of the
time varying mean and spectra from fits performed using AdaptSPEC. These
functions can take a lot of time and memory, so consider thinning the input
\code{x} using \code{\link{window.adaptspecfit}} prior to calling, or
using the \code{time_step} argument.
}
\section{Functions}{
\itemize{
\item \code{time_varying_mean_mean.adaptspecfit}: Posterior mean of the time
varying mean from an adaptspecfit object. Returns a vector.

\item \code{time_varying_mean_samples.adaptspecfit}: Samples of the time varying
mean from an adaptspecfit object. Returns a matrix. Its first dimension is
sample, its second dimension is time. The attribute 'times' contain the
corresponding times (use \code{str()} to inspect this object to see).

\item \code{time_varying_spectra_mean.adaptspecfit}: Posterior mean estimate of
the time varying spectral density from an adaptspecfit object. Returns a
numeric matrix where the rows hold frequencies, columns hold times. The
attributes 'frequencies' and 'times' contain the corresponding
frequencies/times.

\item \code{time_varying_spectra_samples.adaptspecfit}: Samples of the time varying
spectral density from an adaptspecfit object. Returns a three dimensional
array. First dimension is sample, second dimension is frequency, and third
dimension is time. Attributes 'frequencies' and 'times' contain the
corresponding frequencies/times (use \code{str()} to inspect this object to
see).
}}

\examples{
data(simulated_piecewise)
fit <- adaptspec(
  5000,
  1000,
  simulated_piecewise,
  n_bases = 15
)
fit_thin <- window(fit, thin = 10)
tvmm <- time_varying_mean_mean(fit_thin, time_step = 5)
tvsm <- time_varying_spectra_mean(
  fit_thin,
  n_frequencies = 128,
  time_step = 5
)
par(mfrow = c(1, 2))
plot(attr(tvmm, 'times'), tvmm, type = 'l')
image(
  t(tvsm),
  x = attr(tvsm, 'times'),
  y = attr(tvsm, 'frequencies'),
  xlab = 'Time',
  ylab = 'Frequency',
  col = terrain.colors(50)
)
}
