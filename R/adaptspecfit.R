adaptspecfit <- function(results) {
  if (!is.null(results$detrend)) {
    if (results$detrend && length(results$x_missing) > 0) {
      results$x_missing <- lapply(1 : length(results$x_missing), function(i) {
        x_missing <- results$x_missing[[i]]
        if (is.null(x_missing)) return(x_missing)

        missing_indices <- results$missing_indices[[i]]
        x_base <- predict(results$detrend_fits[[i]], data.frame(
          data0 = missing_indices
        ))

        x_missing + x_base
      })
    }
  }

  class(results) <- 'adaptspecfit'

  return(results)
}

#' @export
window.adaptspecfit <- function(fit, ...) {
  fit$n_segments <- window(fit$n_segments, ...)
  fit$tau_squared <- window(fit$tau_squared, ...)
  fit$cut_points <- window(fit$cut_points, ...)
  if (!is.null(fit$log_posterior)) {
    fit$log_posterior <- window(fit$log_posterior, ...)
  }
  fit$beta <- window(fit$beta, ...)
  fit$x_missing <- lapply(fit$x_missing, function(x_missing) {
    if (is.null(x_missing)) {
      x_missing
    } else {
      window(x_missing, ...)
    }
  })
  fit
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

#' @export
summary.adaptspecfit <- function(fit, iterations_threshold = 0) {
  cat('Posterior distribution of number of segments =')
  print(table(fit$n_segments) / length(fit$n_segments))

  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'cut_points'))

  for (n_segments in sort(unique(fit_lcm$n_segments))) {
    if (n_segments == 1) next
    n_iterations <- sum(fit_lcm$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf(
      '--- For n_segments = %d, number of iterations = %d\n',
      n_segments,
      n_iterations
    ))
    for (segment in 1 : (n_segments - 1)) {
      cat('Posterior distribution of cut point', segment, '\n')
      print(table(
        fit_lcm$cut_points[fit_lcm$n_segments == n_segments, segment]
      ) / n_iterations)
    }
  }
}

#' @export
diagnostic_plots.adaptspecfit <- function(fit, iterations_threshold = 0) {
  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'beta'))
  n_iterations <- length(fit_lcm$n_segments)

  data <- do.call(rbind, lapply(
    sort(unique(fit_lcm$n_segments)),
    function(n_segments) {
      indices <- which(fit_lcm$n_segments == n_segments)

      do.call(rbind, lapply(1 : n_segments, function(segment) {
        value <- rep(NA, n_iterations)
        value[indices] <- fit_lcm$beta[indices, segment, 1]
        data.frame(
          iteration = as.vector(time(fit_lcm$n_segments)),
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

#' @export
diagnostics.adaptspecfit <- function(fit, iterations_threshold = 0) {
  cat(sprintf(
    'Tuning parameters: var_inflate = %f, prob_mm1 = %f\n',
    fit$var_inflate,
    fit$prob_mm1
  ))
  cat('Rejection rates for spline fit parameters\n')

  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'beta'))

  for (n_segments in sort(unique(fit_lcm$n_segments))) {
    n_iterations <- sum(fit_lcm$n_segments == n_segments)
    if (n_iterations < iterations_threshold) next
    cat(sprintf(
      '--- For n_segments = %d, number of iterations = %d\n',
      n_segments,
      n_iterations
    ))
    if (n_iterations == 1) next

    rejection_rates <- sapply(1 : n_segments, function(segment) {
      coda::rejectionRate(coda::mcmc(
        fit_lcm$beta[fit_lcm$n_segments == n_segments, segment, 1]
      ))
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
  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'beta', 'tau_squared'))

  n_iterations_total <- length(fit_lcm$n_segments)

  for (n_segments in as.integer(sort(unique(fit_lcm$n_segments)))) {
    n_iterations <- sum(fit_lcm$n_segments == n_segments)

    if (n_iterations / n_iterations_total < iterations_proportion_threshold) {
      next
    }

    for (segment in 1 : n_segments) {
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

  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'cut_points'))

  spec_hat <- segment_spectra_mean(fit)

  for (n_segments in unique(fit_lcm$n_segments)) {
    cut_points <- fit_lcm$cut_points[
      fit_lcm$n_segments == n_segments,
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

#' @export
segment_spectra_mean <- function(fit, n_frequencies = 64) {
  # Compute fits of the spectra
  frequencies <- (0 : (n_frequencies - 1)) / (2 * (n_frequencies - 1))
  nu <- splines_basis1d_demmler_reinsch(frequencies, fit$prior$n_bases)

  fit_lcm <- .thin_to_lcm(fit, c('beta', 'n_segments'))

  spectrum <- list()
  for (n_segments in unique(fit_lcm$n_segments)) {
    spectrum[[n_segments]] <- matrix(
      0, nrow = n_frequencies, ncol = n_segments
    )
    for (segment in 1 : n_segments) {
      beta <- fit_lcm$beta[
        fit_lcm$n_segments == n_segments,
        segment, , drop = FALSE
      ]
      dim(beta) <- dim(beta)[c(1, 3)]
      spectrum[[n_segments]][, segment] <- rowMeans(nu %*% t(beta))
    }
  }

  list(
    frequencies = frequencies,
    spectrum = spectrum
  )
}

#' @export
time_varying_spectra_samples.adaptspecfit <- function(
  fit,
  n_frequencies,
  time_step = 1
) {
  fit_lcm <- .thin_to_lcm(fit, c('n_segments', 'cut_points', 'beta'))
  output <- .time_varying_spectra_samples(
    fit_lcm$n_segments, fit_lcm$cut_points, fit_lcm$beta,
    n_frequencies, time_step
  )
  attr(output, 'times') <- (
    1 + (0 : (dim(output)[3] - 1)) * time_step
  )
  output
}

#' @export
time_varying_spectra_mean.adaptspecfit <- function(
  fit,
  n_frequencies,
  time_step = 1
) {
  samples <- time_varying_spectra_samples(fit, n_frequencies, time_step)
  output <- apply(samples, 2 : 3, mean)
  attr(output, 'frequencies') <- attr(samples, 'frequencies')
  attr(output, 'times') <- (
    1 + (0 : (dim(output)[2] - 1)) * time_step
  )
  output
}

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

    do.call(rbind, lapply(1 : n_segments, function(segment) {
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

.merge_samples.adaptspecfit <- function(x, fits) {
  output <- .merge_mcmc_parts(fits[[1]], fits, c(
    'n_segments',
    'tau_squared',
    'cut_points',
    'log_posterior',
    'beta'
  ))
  output$x_missing <- .merge_x_missing(fits)
  output
}
