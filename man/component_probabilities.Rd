% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adaptspecx.R
\name{component_probabilities}
\alias{component_probabilities}
\title{Calculate mixture probabilities for each time series for AdaptSPEC-X}
\usage{
component_probabilities(x)
}
\arguments{
\item{x}{The object to calculate the probabilities.}
}
\value{
An array with dimensions (iteration, time series, component)
}
\description{
This function calculates samples from the probability of component
membership for each time series for a fit performed by
\code{\link{adaptspec}}.
}
