adaptspecfit <- function(results, n_freq_hat = 0) {
  results$n_segments <- coda::mcmc(results$n_segments)
  results$beta <- aperm(results$beta, c(3, 1, 2))
  results$tau_squared <- coda::mcmc(aperm(results$tau_squared, c(2, 1)))
  results$cut_points <- coda::mcmc(aperm(results$cut_points, c(2, 1)))
  results$log_posterior <- coda::mcmc(results$log_posterior)

  if (n_freq_hat > 0) {
    # Compute fits of the spectra
    freq_hat <- (0 : (n_freq_hat - 1)) / (2 * (n_freq_hat - 1))
    nu_hat <- splines_basis1d_demmler_reinsch(freq_hat, results$prior$n_bases)

    spec_hat <- list()
    for (n_segments in unique(results$n_segments)) {
      spec_hat[[n_segments]] <- matrix(
        0, nrow = n_freq_hat, ncol = n_segments
      )
      for (segment in 1 : n_segments) {
        beta <- results$beta[
          results$n_segments == n_segments,
          segment, , drop = FALSE  # nolint
        ]
        dim(beta) <- dim(beta)[c(1, 3)]
        spec_hat[[n_segments]][, segment] <- rowMeans(nu_hat %*% t(beta))
      }
    }
    results$freq_hat <- freq_hat
    results$spec_hat <- spec_hat
  }

  class(results) <- 'adaptspecfit'

  return(results)
}

#' @export
window.adaptspecfit <- function(fit, ...) {
  time_before <- time(fit$n_segments)
  fit$n_segments <- window(fit$n_segments, ...)
  fit$tau_squared <- window(fit$tau_squared, ...)
  fit$cut_points <- window(fit$cut_points, ...)
  fit$log_posterior <- window(fit$log_posterior, ...)

  time_after <- time(fit$n_segments)
  fit$beta <- fit$beta[time_before %in% time_after, , ]

  fit
}

#' @export
summary.adaptspecfit <- function(fit, iterations_threshold = 0) {
  cat('Posterior distribution of number of segments =')
  print(table(fit$n_segments) / length(fit$n_segments))

  for (n_segments in sort(unique(fit$n_segments))) {
    if (n_segments == 1) next
    n_iterations <- sum(fit$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf('--- For n_segments = %d, number of iterations = %d\n', n_segments, n_iterations))
    for (segment in 1 : (n_segments - 1)) {
      cat('Posterior distribution of cut point', segment, '\n')
      print(table(
        fit$cut_points[fit$n_segments == n_segments, segment]
      ) / n_iterations)
    }
  }
}

#' @export
diagnostic_plots.adaptspecfit <- function(fit, iterations_threshold = 0) {
  n_iterations <- length(fit$n_segments)

  data <- do.call(rbind, lapply(sort(unique(fit$n_segments)), function(n_segments) {
    indices <- which(fit$n_segments == n_segments)

    do.call(rbind, lapply(1 : n_segments, function(segment) {
      value <- rep(NA, n_iterations)
      value[indices] <- fit$beta[indices, segment, 1]
      data.frame(
        iteration = 1 : n_iterations,
        n_segments = factor(n_segments),
        segment = factor(segment),
        value = value
      )
    }))
  }))

  ggplot2::ggplot(data, ggplot2::aes(iteration, value)) +
    ggplot2::geom_line(na.rm = TRUE) +
    ggplot2::facet_grid(segment ~ n_segments, scales = 'free', labeller = ggplot2::label_both)
}

#' @export
diagnostics.adaptspecfit <- function(fit, iterations_threshold = 0) {
  cat(sprintf('Tuning parameters: var_inflate = %f, prob_mm1 = %f\n', fit$var_inflate, fit$prob_mm1))
  cat('Rejection rates for spline fit parameters\n')
  for (n_segments in sort(unique(fit$n_segments))) {
    n_iterations <- sum(fit$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf('--- For n_segments = %d, number of iterations = %d\n', n_segments, n_iterations))
    if (n_iterations == 1) next

    rejection_rates <- sapply(1 : n_segments, function(segment) {
      coda::rejectionRate(coda::mcmc(fit$beta[fit$n_segments == n_segments, segment, 1]))
    })
    names(rejection_rates) <- 1 : n_segments
    print(rejection_rates)
  }
}

#' @export
diagnostic_warnings.adaptspecfit <- function(
  fit,
  effective_size_threshold = 100,
  iterations_proportion_threshold = 0.01,
  prefix = ''
) {
  n_iterations_total <- length(fit$n_segments)

  for (n_segments in as.integer(sort(unique(fit$n_segments)))) {
    n_iterations <- sum(fit$n_segments == n_segments)

    if (n_iterations / n_iterations_total < iterations_proportion_threshold) next

    for (segment in 1 : n_segments) {
      if (sum(fit$n_segments == n_segments) == 1) next
      beta_worst_neff <- min(coda::effectiveSize(coda::mcmc(
        fit$beta[fit$n_segments == n_segments, segment, ]
      )))
      if (beta_worst_neff <= effective_size_threshold) {
        warning(sprintf(
          '%sn_segments = %d, segment = %d has beta with effective sample size %f',
          prefix, n_segments, segment, beta_worst_neff
        ))
      }

      tau_squared_neff <- coda::effectiveSize(
        fit$tau_squared[fit$n_segments == n_segments, segment]
      )
      if (tau_squared_neff <= effective_size_threshold) {
        warning(sprintf(
          '%sn_segments = %d, segment = %d tau_squared has effective sample size %f',
          prefix, n_segments, segment, tau_squared_neff
        ))
      }
    }
  }

  invisible(NULL)
}

#' @name plot.adaptspecfit
#'
#' @title Summary plots of adaptspecfit objects
#'
#' @description Plots an adaptspecfit object
#'
#' @param fit The fit object, as returned by adaptspec
#' @param ask Prompt user before each page of plots
#' @param auto_layout Automatically generate output format
#'
#' @usage plot.mcmc(fit, ask = dev.interactive(), auto_layout = TRUE)
#'
#' @export
plot.adaptspecfit <- function(fit, ask, auto_layout = TRUE) {
  if (missing(ask)) {
    ask <- dev.interactive()
  }

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  if (auto_layout) {
    par(mfrow = c(2, 2))
  }

  coda::traceplot(fit$n_segments, main = 'Number of segments traceplot')
  par(ask = ask)
  barplot(
    table(fit$n_segments) / length(fit$n_segments),
    main = 'Number of segments histogram',
    ylim = c(0, 1)
  )

  for (n_segments in unique(fit$n_segments)) {
    cut_points <- fit$cut_points[
      fit$n_segments == n_segments,
      1 : n_segments,
      drop = FALSE
    ]
    for (segment in 1 : n_segments) {
      hist(
        cut_points[, segment], prob = TRUE,
        main = sprintf(
          'Segment %d endpoints (%d segments)',
          segment, n_segments
        ),
        xlab = ''
      )

      if (!is.null(fit$freq_hat)) {
        plot(
          fit$freq_hat, fit$spec_hat[[n_segments]][, segment],
          type = 'l', xlab = 'Frequency', ylab = 'Log spectral density',
          main = sprintf(
            'Segment %d log spectral density (%d segments)',
            segment, n_segments
          )
        )
      }
    }
  }
}

#' @export
time_varying_spectra_samples.adaptspecfit <- function(fit, n_frequencies) {
  .time_varying_spectra_samples(fit$n_segments, fit$cut_points, fit$beta, n_frequencies)
}

#' @export
time_varying_spectra_mean.adaptspecfit <- function(fit, n_frequencies) {
  samples <- time_varying_spectra_samples(fit, n_frequencies)
  output <- apply(samples, 2 : 3, mean)
  attr(output, 'frequencies') <- attr(samples, 'frequencies')
  output
}
