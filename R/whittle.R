#' @export
sample_whittle_spline <- function(n, beta) {
  n_frequencies <- floor(n / 2) + 1
  frequencies <- seq(0, 0.5, length.out = n_frequencies)
  nu <- splines_basis1d(frequencies, length(beta) - 1, omitLinear = TRUE)
  f_hat <- nu %*% beta

  # Draw a DFT (so to speak)
  dft_first <- exp(f_hat[1] / 2) * rnorm(1)
  if (n %% 2 == 0) {
    dft_middle <- (sqrt(0.5) * exp(as.vector(f_hat[2 : (n_frequencies - 1)]) / 2)) * (
      rnorm(n_frequencies - 2) - 1i * rnorm(n_frequencies - 2)
    )
    dft_last <- exp(f_hat[n_frequencies] / 2)
  } else {
    dft_middle <- (sqrt(0.5) * exp(as.vector(f_hat[2 : n_frequencies]) / 2)) * (
      rnorm(n_frequencies - 1) - 1i * rnorm(n_frequencies - 1)
    )
    dft_last <- c()
  }
  dft_full <- c(dft_first, dft_middle, dft_last, rev(Conj(dft_middle)))

  return(Re(fft(dft_full, inverse = TRUE) / sqrt(n)))
}
