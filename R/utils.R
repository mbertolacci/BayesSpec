.extend_list <- function(...) {
  lists <- list(...)
  output <- lists[[1]]
  for (value in lists[2 : length(lists)]) {
    for (name in names(value)) {
      output[[name]] <- value[[name]]
    }
  }
  return(output)
}

#' @export
ntimes <- function(x, ...) {
  UseMethod('ntimes', x)
}

#' @export
diagnostics <- function(x, ...) UseMethod('diagnostics', x)

#' @export
diagnostic_plots <- function(x, ...) UseMethod('diagnostic_plots', x)

#' @export
diagnostic_warnings <- function(x, ...) UseMethod('diagnostic_warnings', x)

#' @export
time_varying_mean_samples <- function(x, ...) {
  UseMethod('time_varying_mean_samples', x)
}

#' @export
time_varying_mean_mean <- function(x, ...) {
  UseMethod('time_varying_mean_mean', x)
}

#' @export
time_varying_spectra_samples <- function(x, ...) {
  UseMethod('time_varying_spectra_samples', x)
}

#' @export
time_varying_spectra_mean <- function(x, ...) {
  UseMethod('time_varying_spectra_mean', x)
}

#' @export
component_probabilities <- function(x, ...) {
  UseMethod('component_probabilities', x)
}

.merge_samples <- function(x, ...) {
  UseMethod('.merge_samples', x)
}

#' @export
merge_samples <- function(values) {
  .merge_samples(values[[1]], values)
}

.gcd <- function(a, b) {
  while (b != 0) {
    t <- b
    b <- a %% b
    a <- t
  }
  a
}

.lcm <- function(a, b) {
  as.integer((a * b) / .gcd(a, b))
}

.missing_indices <- function(data) {
  lapply(seq_len(ncol(data)), function(i) which(is.na(data[, i])))
}

.zero_index_missing_indices <- function(missing_indices) {
  lapply(missing_indices, `-`, 1)
}

.prepare_data <- function(data, detrend) {
  data <- as.matrix(data)
  detrend_fits <- NULL
  if (detrend && ncol(data) > 0) {
    # Detrend the observations (nolint because lintr can't figure out this
    # is used below)
    data0 <- 1 : nrow(data)  # nolint
    detrend_fits <- list()
    for (series in 1 : ncol(data)) {
      detrend_fits[[series]] <- lm(
        data[, series] ~ data0,
        na.action = na.exclude
      )
      data[, series] <- residuals(detrend_fits[[series]])
    }
  }

  list(
    data = data,
    detrend_fits = detrend_fits,
    missing_indices = .missing_indices(data)
  )
}

.x_missing_start <- function(start, missing_indices) {
  if (is.null(start$x_missing)) {
    start$x_missing <- lapply(missing_indices, function(x) {
      rnorm(length(x))
    })
  }
  start
}

.validate_x_missing_start <- function(start, missing_indices) {
  for (i in seq_len(length(missing_indices))) {
    stopifnot(length(missing_indices[[i]]) == length(start$x_missing[[i]]))
  }
}

.merge_mcmc <- function(values) {
  if (is.null(values[[1]])) {
    return(NULL)
  }
  if (is.null(dim(values[[1]]))) {
    how <- c
  } else if (length(dim(values[[1]])) < 3) {
    how <- rbind
  } else {
    how <- function(...) abind::abind(..., along = 1)
  }
  output <- do.call(how, values)
  if (is.null(dim(output)) || length(dim(output)) < 3) {
    class(output) <- 'mcmc'
  } else {
    class(output) <- 'mcmca'
  }
  n_output <- NULL
  if (is.null(dim(output))) {
    n_output <- length(output)
  } else {
    n_output <- dim(output)[1]
  }
  attr(output, 'mcpar') <- c(1, n_output, 1)
  output
}

.merge_mcmc_parts <- function(start, values, parts) {
  output <- start
  for (part in parts) {
    output[[part]] <- .merge_mcmc(lapply(values, getElement, part))
  }
  output
}

.merge_x_missing <- function(fits) {
  lapply(seq_len(length(fits[[1]]$x_missing)), function(i) {
    if (is.null(fits[[1]]$x_missing[[i]])) NULL
    else {
      values <- lapply(fits, function(fit) fit$x_missing[[i]])
      .merge_mcmc(values)
    }
  })
}
