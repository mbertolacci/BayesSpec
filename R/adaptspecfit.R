.adaptspecfit <- function(results) {
  if (!is.null(results$detrend)) {
    if (results$detrend && length(results$x_missing) > 0) {
      results$x_missing <- lapply(
        seq_len(length(results$x_missing)),
        function(i) {
          x_missing <- results$x_missing[[i]]
          if (is.null(x_missing)) return(x_missing)

          missing_indices <- results$missing_indices[[i]]
          x_base <- stats::predict(results$detrend_fits[[i]], data.frame(
            data0 = missing_indices
          ))

          x_missing + x_base
        }
      )
    }
  }

  class(results) <- 'adaptspecfit'

  return(results)
}

#' @name adaptspecfit
#' @title Methods for adaptspecfit objects
#' @description These methods apply to the adaptspecfit objects returned by
#' \code{\link{adaptspec}}.
#' @param x,object \code{adaptspecfit} object
#' @param iterations_threshold Number of iterations below which to suppress
#' output. For example, if this is 100, and the MCMC chain spent less than
#' 100 iterations in, say, n_segments = 4, summary output will not include
#' n_segments = 4
#' @param effective_size_threshold Minimum effective sample size (see
#' \code{\link[coda]{effectiveSize}}) below which a warning is output.
#' @param iterations_proportion_threshold Proportion of iterations below which
#' to suppress output. Just like \code{iterations_threshold}, but a proportion.
#' @param prefix Used internally.
#' @param ask Prompt user before each page of plots.
#' @param auto_layout Automatically generate output format.
#' @param ... Extra arguments. For \code{window}, these are passed to
#' \code{\link[coda]{window.mcmc}} to set the \code{thin} or \code{start}
#' parameters (for example)
#' @seealso \code{\link{segment_log_spectra_mean}} for methods calculating
#' estimates of the spectral densities in each segments.
#' \code{\link{time_varying_mean_mean.adaptspecfit}} for estimates of the
#' time-varying mean and spectral density.
NULL

#' @describeIn adaptspecfit Method to modify the start/thinning of MCMC samples,
#' as per \code{\link[coda]{window.mcmc}}
#' @export
window.adaptspecfit <- function(x, ...) {
  x$n_segments <- window(x$n_segments, ...)
  x$tau_squared <- window(x$tau_squared, ...)
  x$cut_points <- window(x$cut_points, ...)
  if (!is.null(x$log_posterior)) {
    x$log_posterior <- window(x$log_posterior, ...)
  }
  x$beta <- window(x$beta, ...)
  x$mu <- window(x$mu, ...)
  x$x_missing <- lapply(x$x_missing, function(x_missing) {
    if (is.null(x_missing)) {
      x_missing
    } else {
      window(x_missing, ...)
    }
  })
  x
}

.thin_to_lcm <- function(fit, var_names) {
  var_lcm <- Reduce(function(lcm, var_name) {
    .lcm(thin(fit[[var_name]]), lcm)
  }, var_names, 1)  # nolint
  for (var_name in var_names) {
    fit[[var_name]] <- window(fit[[var_name]], thin = var_lcm)
  }
  fit
}

#' @describeIn adaptspecfit Summarises MCMC samples.
#' @export
summary.adaptspecfit <- function(object, iterations_threshold = 0, ...) {
  cat('Posterior distribution of number of segments =')
  print(table(object$n_segments) / length(object$n_segments))

  fit_lcm <- .thin_to_lcm(object, c('n_segments', 'cut_points'))

  for (n_segments in sort(unique(fit_lcm$n_segments))) {
    if (n_segments == 1) next
    n_iterations <- sum(fit_lcm$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf(
      '--- For n_segments = %d, number of iterations = %d\n',
      n_segments,
      n_iterations
    ))
    for (segment in seq_len(n_segments - 1)) {
      cat('Posterior distribution of cut point', segment, '\n')
      print(table(
        fit_lcm$cut_points[fit_lcm$n_segments == n_segments, segment]
      ) / n_iterations)
    }
  }
}

#' @describeIn adaptspecfit Returns the number of time periods in the input
#' @export
ntimes.adaptspecfit <- function(x, ...) {
  x$cut_points[1, ncol(x$cut_points)]
}

#' @describeIn adaptspecfit MCMC diagnostic plots to assess convergence.
#' @export
diagnostic_plots.adaptspecfit <- function(x, ...) {
  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'beta', 'tau_squared'))
  n_iterations <- length(fit_lcm$n_segments)

  plot_traces <- function(beta_i) {
    data <- do.call(rbind, lapply(
      sort(unique(fit_lcm$n_segments)),
      function(n_segments) {
        indices <- which(fit_lcm$n_segments == n_segments)

        do.call(rbind, lapply(seq_len(n_segments), function(segment) {
          value <- rep(NA, n_iterations)
          if (!is.null(beta_i)) {
            value[indices] <- fit_lcm$beta[indices, segment, beta_i]
          } else {
            value[indices] <- fit_lcm$tau_squared[indices, segment]
          }
          data.frame(
            iteration = as.vector(stats::time(fit_lcm$n_segments)),
            n_segments = factor(n_segments),
            segment = factor(segment),
            value = value
          )
        }))
      })
    )

    ggplot2::ggplot(data, ggplot2::aes(iteration, value)) +
      ggplot2::geom_line(na.rm = TRUE) +
      ggplot2::facet_grid(
        segment ~ n_segments,
        scales = 'free',
        labeller = ggplot2::label_both
      )
  }

  gridExtra::grid.arrange(
    plot_traces(1) + ggplot2::ggtitle('beta[1]'),
    plot_traces(2) + ggplot2::ggtitle('beta[2]'),
    plot_traces(3) + ggplot2::ggtitle('beta[3]'),
    plot_traces(NULL) + ggplot2::ggtitle('tau_squared'),
    ncol = 4,
    ...
  )
}

#' @describeIn adaptspecfit Outputs MCMC diagnostics statistics to help assess
#' convergence.
#' @export
diagnostics.adaptspecfit <- function(x, iterations_threshold = 0, ...) {
  cat(sprintf(
    'Tuning parameters: var_inflate = %f, prob_mm1 = %f\n',
    x$var_inflate,
    x$prob_mm1
  ))
  cat('Rejection rates for spline fit parameters\n')

  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'beta'))

  for (n_segments in sort(unique(fit_lcm$n_segments))) {
    n_iterations <- sum(fit_lcm$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf(
      '--- For n_segments = %d, number of iterations = %d\n',
      n_segments,
      n_iterations
    ))
    if (n_iterations == 1) next

    rejection_rates <- sapply(seq_len(n_segments), function(segment) {
      coda::rejectionRate(coda::mcmc(
        fit_lcm$beta[fit_lcm$n_segments == n_segments, segment, 1]
      ))
    })
    names(rejection_rates) <- seq_len(n_segments)
    print(rejection_rates)
  }
}

#' @describeIn adaptspecfit Outputs warnings when MCMC diagnostics are below
#' outside of nominal threshold ranges. The user is cautioned that this is
#' subject to both false positive and false negatives; examining diagnostics
#' plots directly is advised.
#' @export
diagnostic_warnings.adaptspecfit <- function(
  x,
  effective_size_threshold = 100,
  iterations_proportion_threshold = 0.01,
  prefix = '',
  ...
) {
  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'beta', 'tau_squared'))

  n_iterations_total <- length(fit_lcm$n_segments)

  for (n_segments in as.integer(sort(unique(fit_lcm$n_segments)))) {
    n_iterations <- sum(fit_lcm$n_segments == n_segments)

    if (n_iterations / n_iterations_total < iterations_proportion_threshold) {
      next
    }

    for (segment in seq_len(n_segments)) {
      if (sum(fit_lcm$n_segments == n_segments) == 1) next
      beta_worst_neff <- min(coda::effectiveSize(coda::mcmc(
        fit_lcm$beta[fit_lcm$n_segments == n_segments, segment, ]
      )))
      if (beta_worst_neff <= effective_size_threshold) {
        warning(sprintf(
          paste0(
            '%sn_segments = %d, segment = %d has beta with effective sample ',
            'size %f'
          ),
          prefix, n_segments, segment, beta_worst_neff
        ))
      }

      tau_squared_neff <- coda::effectiveSize(
        fit_lcm$tau_squared[fit_lcm$n_segments == n_segments, segment]
      )
      if (tau_squared_neff <= effective_size_threshold) {
        warning(sprintf(
          paste0(
            '%sn_segments = %d, segment = %d tau_squared has effective sample ',
            'size %f'
          ),
          prefix, n_segments, segment, tau_squared_neff
        ))
      }
    }
  }

  invisible(NULL)
}

#' @describeIn adaptspecfit Summary plots
#' @export
plot.adaptspecfit <- function(x, ask, auto_layout = TRUE, ...) {
  if (missing(ask)) {
    ask <- grDevices::dev.interactive()
  }

  old_par <- grDevices::par(no.readonly = TRUE)
  on.exit(grDevices::par(old_par))
  if (auto_layout) {
    grDevices::par(mfrow = c(2, 2))
  }

  coda::traceplot(x$n_segments, main = 'Number of segments traceplot')
  par(ask = ask)
  graphics::barplot(
    table(x$n_segments) / length(x$n_segments),
    main = 'Number of segments histogram',
    ylim = c(0, 1)
  )

  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'cut_points'))

  spec_hat <- segment_spectra_mean(x)

  for (n_segments in unique(fit_lcm$n_segments)) {
    cut_points <- fit_lcm$cut_points[
      fit_lcm$n_segments == n_segments,
      seq_len(n_segments),
      drop = FALSE
    ]
    for (segment in seq_len(n_segments)) {
      graphics::hist(
        cut_points[, segment], prob = TRUE,
        main = sprintf(
          'Segment %d endpoints (%d segments)',
          segment, n_segments
        ),
        xlab = ''
      )

      plot(
        spec_hat$frequencies, spec_hat$spectrum[[n_segments]][, segment],
        type = 'l', xlab = 'Frequency', ylab = 'Log spectral density',
        main = sprintf(
          'Segment %d log spectral density (%d segments)',
          segment, n_segments
        )
      )
    }
  }
}

#' Estimates of segment spectral densities from adaptspecfit objects
#'
#' These methods calculates estimates of the segment spectral densities
#' modelled by \code{\link{adaptspec}}, as evaluated at specified frequencies.
#' \code{segment_log_spectra_mean} calculates the posterior mean of the log
#' spectral density, while \code{segment_spectra_mean} does the same for the
#' spectral density.
#'
#' @param fit \code{adaptspecfit} object
#' @param n_frequencies Number of frequencies at which to evaluate the spectral
#' densities
#' @param frequencies Frequencies at which to evaluate the spectral density.
#' Should be between 0 and 0.5.
#' @return For \code{segment_log_spectra_mean}, a list with entries
#' \code{frequencies} and \code{log_spectrum}. For \code{segment_spectra_mean},
#' the entries are \code{frequencies} and \code{spectrum}.
#' @export
segment_log_spectra_mean <- function(
  fit,
  n_frequencies = 64,
  frequencies = seq(0, 0.5, length.out = n_frequencies)
) {
  if (missing(n_frequencies) && !missing(frequencies)) {
    n_frequencies <- length(frequencies)
  }
  # Compute fits of the spectra
  if (fit$prior$frequency_transform == 'cbrt') {
    transformed_frequencies <- (2 * frequencies) ^ (1 / 3)
  } else {
    transformed_frequencies <- 2 * frequencies
  }
  nu <- splines_basis1d_demmler_reinsch(
    transformed_frequencies,
    fit$prior$n_bases
  )

  fit_lcm <- .thin_to_lcm(fit, c('beta', 'n_segments'))

  log_spectrum <- list()
  for (n_segments in unique(fit_lcm$n_segments)) {
    log_spectrum[[n_segments]] <- matrix(
      0, nrow = n_frequencies, ncol = n_segments
    )
    for (segment in seq_len(n_segments)) {
      beta <- fit_lcm$beta[
        fit_lcm$n_segments == n_segments,
        segment, , drop = FALSE
      ]
      dim(beta) <- dim(beta)[c(1, 3)]
      log_spectrum[[n_segments]][, segment] <- rowMeans(nu %*% t(beta))
    }
  }

  list(
    frequencies = frequencies,
    log_spectrum = log_spectrum
  )
}

#' @describeIn segment_log_spectra_mean Posterior mean of spectral density in
#' each segment.
#' @export
segment_spectra_mean <- function(
  fit,
  n_frequencies = 64,
  frequencies = seq(0, 0.5, length.out = n_frequencies)
) {
  if (missing(n_frequencies) && !missing(frequencies)) {
    n_frequencies <- length(frequencies)
  }
  # Compute fits of the spectra
  if (fit$prior$frequency_transform == 'cbrt') {
    transformed_frequencies <- (2 * frequencies) ^ (1 / 3)
  } else {
    transformed_frequencies <-  2 * frequencies
  }
  nu <- splines_basis1d_demmler_reinsch(
    transformed_frequencies,
    fit$prior$n_bases
  )

  fit_lcm <- .thin_to_lcm(fit, c('beta', 'n_segments'))

  spectrum <- list()
  for (n_segments in unique(fit_lcm$n_segments)) {
    spectrum[[n_segments]] <- matrix(
      0, nrow = n_frequencies, ncol = n_segments
    )
    for (segment in seq_len(n_segments)) {
      beta <- fit_lcm$beta[
        fit_lcm$n_segments == n_segments,
        segment, , drop = FALSE
      ]
      dim(beta) <- dim(beta)[c(1, 3)]
      spectrum[[n_segments]][, segment] <- rowMeans(exp(
        nu %*% t(beta)
      ))
    }
  }

  list(
    frequencies = frequencies,
    spectrum = spectrum
  )
}

#' @name time_varying_mean_mean.adaptspecfit
#' @title Posterior estimates of the time varying mean and spectra from an
#' adaptspecfit object
#'
#' @description These methods calculate posterior means or samples of the
#' time varying mean and spectra from fits performed using AdaptSPEC. These
#' functions can take a lot of time and memory, so consider thinning the input
#' \code{x} using \code{\link{window.adaptspecfit}} prior to calling, or
#' using the \code{time_step} argument.
#'
#' @param x \code{adaptspecfit} object
#' @param time_step Time varying quantity is calculated only at times
#' divisible by this number. Reduces the size of the output. Ignore if
#' \code{times} is provided.
#' @param times Times at which to calculate the quantity. Must be between
#' 1 and \code{ntimes(x)}, inclusive.
#' @param n_frequencies Number of frequencies at which to evaluate the spectral
#' densities. Ignored if \code{frequencies} is set.
#' @param frequencies Frequencies at which to evaluate the spectral density.
#' Must be between 0 and 0.5, inclusive.
#' @param ... Ignored.
#' @examples
#' data(simulated_piecewise)
#' fit <- adaptspec(
#'   5000,
#'   1000,
#'   simulated_piecewise,
#'   n_bases = 15
#' )
#' fit_thin <- window(fit, thin = 10)
#' tvmm <- time_varying_mean_mean(fit_thin, time_step = 5)
#' tvsm <- time_varying_spectra_mean(
#'   fit_thin,
#'   n_frequencies = 128,
#'   time_step = 5
#' )
#' par(mfrow = c(1, 2))
#' plot(attr(tvmm, 'times'), tvmm, type = 'l')
#' image(
#'   t(tvsm),
#'   x = attr(tvsm, 'times'),
#'   y = attr(tvsm, 'frequencies'),
#'   xlab = 'Time',
#'   ylab = 'Frequency',
#'   col = terrain.colors(50)
#' )
NULL

#' @describeIn time_varying_mean_mean.adaptspecfit Posterior mean of the time
#' varying mean from an adaptspecfit object. Returns a vector.
#' @export
time_varying_mean_mean.adaptspecfit <- function(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  samples <- time_varying_mean_samples(x, times = times)
  output <- colMeans(samples)
  attr(output, 'times') <- attr(samples, 'times')
  output
}

#' @describeIn time_varying_mean_mean.adaptspecfit Samples of the time varying
#' mean from an adaptspecfit object. Returns a matrix. Its first dimension is
#' sample, its second dimension is time. The attribute 'times' contain the
#' corresponding times (use \code{str()} to inspect this object to see).
#' @export
time_varying_mean_samples.adaptspecfit <- function(
  x,
  time_step = 1,
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'cut_points', 'mu'))
  output <- .time_varying_mean_samples(
    fit_lcm$n_segments,
    fit_lcm$cut_points,
    fit_lcm$mu,
    times
  )
  output
}

#' @describeIn time_varying_mean_mean.adaptspecfit Posterior mean estimate of
#' the time varying spectral density from an adaptspecfit object. Returns a
#' numeric matrix where the rows hold frequencies, columns hold times. The
#' attributes 'frequencies' and 'times' contain the corresponding
#' frequencies/times.
#' @export
time_varying_spectra_mean.adaptspecfit <- function(
  x,
  n_frequencies = 64,
  time_step = 1,
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  samples <- time_varying_spectra_samples(
    x,
    frequencies = frequencies,
    times = times
  )
  output <- apply(samples, 2 : 3, mean)
  attr(output, 'frequencies') <- attr(samples, 'frequencies')
  attr(output, 'times') <- attr(samples, 'times')
  output
}

#' @describeIn time_varying_mean_mean.adaptspecfit Samples of the time varying
#' spectral density from an adaptspecfit object. Returns a three dimensional
#' array. First dimension is sample, second dimension is frequency, and third
#' dimension is time. Attributes 'frequencies' and 'times' contain the
#' corresponding frequencies/times (use \code{str()} to inspect this object to
#' see).
#' @export
time_varying_spectra_samples.adaptspecfit <- function(
  x,
  n_frequencies = 64,
  time_step = 1,
  frequencies = seq(0, 0.5, length.out = n_frequencies),
  times = seq(1, ntimes(x), by = time_step),
  ...
) {
  stopifnot(min(times) >= 1 || max(times) <= ntimes(x))
  stopifnot(min(frequencies) >= 0 && max(frequencies) <= 0.5)
  fit_lcm <- .thin_to_lcm(x, c('n_segments', 'cut_points', 'beta'))
  output <- .time_varying_spectra_samples(
    fit_lcm$n_segments, fit_lcm$cut_points, fit_lcm$beta,
    frequencies, times, x$prior$frequency_transform
  )
  output
}

#' Calculates the estimated posterior PMF of the cut points
#'
#' Returns a data frame with four columns:
#' \itemize{
#'   \item n_segments Number of segments
#'   \item segment Segment index
#'   \item cut_point Value of the cutpoint (i.e., time of end of segment)
#'   \item probability Estimated probability that the cut point for this
#'   segment is this value when the number of segments is n_segments
#' }
#'
#' @param fit \code{adaptspecfit} object
#' @param within_n_segments Whether to normalise the probabilities so they
#' sum to one within each value of n_segments. If FALSE, summing down the
#' probability column will be equal to one overall.
#' @export
cut_point_pmf <- function(fit, within_n_segments = FALSE) {
  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'cut_points'))
  n_iterations <- length(fit_lcm$n_segments)
  max_index <- max(fit_lcm$cut_points)
  do.call(rbind, lapply(unique(fit_lcm$n_segments), function(n_segments) {
    cut_points <- fit_lcm$cut_points[
      fit_lcm$n_segments == n_segments,
      ,
      drop = FALSE
    ]
    n_normalise <- n_iterations
    if (within_n_segments) {
      n_normalise <- nrow(cut_points)
    }

    do.call(rbind, lapply(seq_len(n_segments), function(segment) {
      counts <- tabulate(cut_points[, segment], max_index)
      probabilities <- counts / n_normalise
      non_zero <- probabilities > 0
      data.frame(
        n_segments = n_segments,
        segment = segment,
        cut_point = which(non_zero),
        probability = probabilities[non_zero]
      )
    }))
  }))
}

.merge_samples.adaptspecfit <- function(x, fits) {  # nolint
  output <- .merge_mcmc_parts(fits[[1]], fits, c(
    'n_segments',
    'tau_squared',
    'cut_points',
    'log_posterior',
    'beta',
    'mu'
  ))
  output$x_missing <- .merge_x_missing(fits)
  output
}
