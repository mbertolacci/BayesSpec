% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adaptspecfit.R
\name{cut_point_pmf}
\alias{cut_point_pmf}
\title{Calculates the estimated posterior PMF of the cut points}
\usage{
cut_point_pmf(fit, within_n_segments = FALSE)
}
\arguments{
\item{fit}{\code{adaptspecfit} object}

\item{within_n_segments}{Whether to normalise the probabilities so they
sum to one within each value of n_segments. If FALSE, summing down the
probability column will be equal to one overall.}
}
\description{
Returns a data frame with four columns:
\itemize{
  \item n_segments Number of segments
  \item segment Segment index
  \item cut_point Value of the cutpoint (i.e., time of end of segment)
  \item probability Estimated probability that the cut point for this
  segment is this value when the number of segments is n_segments
}
}
