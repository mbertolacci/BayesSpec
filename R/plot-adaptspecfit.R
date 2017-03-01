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
    par(mfrow = c(4, 2))
  }

  coda::traceplot(fit$n_segments, main = 'Number of segments traceplot')
  par(ask = ask)
  coda::densplot(fit$n_segments, main = 'Number of segments histogram')

  for (n_segments in unique(results$n_segments)) {
    cut_point <- fit$cut_point[
      fit$n_segments == n_segments,
      1 : n_segments,
      drop = FALSE
    ]
    for (segment in 1 : n_segments) {
      hist(
        cut_point[, segment], prob = TRUE,
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
