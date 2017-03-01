context('sample')

x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

prior <- list(
  n_segments_max = 3,
  t_min = 1,
  sigma_squared_alpha = 1,
  tau_prior_a = 1,
  tau_prior_b = 1,
  tau_upper_limit = 1,
  n_bases = 2
)

test_that('initialise', {
  result <- .get_sample_default(x, prior, 1)
  expect_equal(result$cut_points, c(20, 20, 20))
  expect_equal(result$segment_lengths, c(20, 0, 0))

  result <- .get_sample_default(x, prior, 2)
  expect_equal(result$cut_points, c(10, 20, 20))
  expect_equal(result$segment_lengths, c(10, 10, 0))

  result <- .get_sample_default(x, prior, 3)
  expect_equal(result$cut_points, c(6, 13, 20))
  expect_equal(result$segment_lengths, c(6, 7, 7))
})

test_that('fit and densities', {
  result <- .get_sample_filled(
    x, prior,
    1,
    matrix(0, nrow = 3, ncol = 3),
    rep(1, 3),
    c(20, 20, 20)
  )

  # Basis
  expect_equal(result$periodogram[[1]], matrix(c(
    0.000000000, 4.699544030, 0.026898256, 0.008475845, 0.004491217,
    0.002946380, 0.002191168, 0.001779383, 0.001548366, 0.001429210,
    0.001392288
  ), ncol = 1))
  expect_equal(result$nu[[1]], rbind(
    c(1,  2.250791e-01,  0.11253954),
    c(1,  2.140629e-01,  0.09104640),
    c(1,  1.820928e-01,  0.03477663),
    c(1,  1.322982e-01, -0.03477663),
    c(1,  6.955326e-02, -0.09104640),
    c(1,  1.378212e-17, -0.11253954),
    c(1, -6.955326e-02, -0.09104640),
    c(1, -1.322982e-01, -0.03477663),
    c(1, -1.820928e-01,  0.03477663),
    c(1, -2.140629e-01,  0.09104640),
    c(1, -2.250791e-01,  0.11253954)
  ), tolerance = 1e-5)

  # Likelihood and prior
  expect_equal(result$log_segment_likelihood[1], -23.12877, tolerance = 1e-5)
  expect_equal(result$log_segment_prior[1], -2.756816, tolerance = 1e-5)

  # Proposal distribution
  expect_equal(result$log_segment_proposal[1], -6.060031, tolerance = 1e-5)
  expect_equal(result$beta_mle[1, ], c(-1.1004098, 1.8866003, 0.7976257))
  expect_equal(result$precision_cholesky_mle[[1]], rbind(
    c(3.146361, 0.5996133, 0.25350734),
    c(0.000000, 1.0224588, 0.01854398),
    c(0.000000, 0.0000000, 1.00428261)
  ), tolerance = 1e-5)

  # Multivariate fit
  result <- .get_sample_filled(
    cbind(x, x), prior,
    1,
    matrix(0, nrow = 3, ncol = 3),
    rep(1, 3),
    c(20, 20, 20)
  )

  # Basis
  expect_equal(result$periodogram[[1]], cbind(c(
    0.000000000, 4.699544030, 0.026898256, 0.008475845, 0.004491217,
    0.002946380, 0.002191168, 0.001779383, 0.001548366, 0.001429210,
    0.001392288
  ), c(
    0.000000000, 4.699544030, 0.026898256, 0.008475845, 0.004491217,
    0.002946380, 0.002191168, 0.001779383, 0.001548366, 0.001429210,
    0.001392288
  )))
  expect_equal(result$nu[[1]], rbind(
    c(1,  2.250791e-01,  0.11253954),
    c(1,  2.140629e-01,  0.09104640),
    c(1,  1.820928e-01,  0.03477663),
    c(1,  1.322982e-01, -0.03477663),
    c(1,  6.955326e-02, -0.09104640),
    c(1,  1.378212e-17, -0.11253954),
    c(1, -6.955326e-02, -0.09104640),
    c(1, -1.322982e-01, -0.03477663),
    c(1, -1.820928e-01,  0.03477663),
    c(1, -2.140629e-01,  0.09104640),
    c(1, -2.250791e-01,  0.11253954)
  ), tolerance = 1e-5)

  # Likelihood and prior
  expect_equal(result$log_segment_likelihood[1], -46.25754, tolerance = 1e-5)
  expect_equal(result$log_segment_prior[1], -2.756816, tolerance = 1e-5)

  # Proposal distribution
  expect_equal(result$log_segment_proposal[1], -15.65005, tolerance = 1e-5)
  expect_equal(
    result$beta_mle[1, ],
    c(-1.623733, 3.860757, 1.632835),
    tolerance = 1e-5
  )
  expect_equal(result$precision_cholesky_mle[[1]], rbind(
    c(4.401848, 0.8770763, 0.37094314),
    c(0.000000, 1.0313731, 0.02388138),
    c(0.000000, 0.0000000, 1.00630540)
  ), tolerance = 1e-5)
})

test_that('log_prior_cut_points', {
  result <- .get_sample_filled(
    x, prior,
    1,
    matrix(0, nrow = 3, ncol = 3),
    rep(1, 3),
    c(20, 20, 20)
  )
  expect_equal(result$log_prior_cut_points, 0)

  result <- .get_sample_filled(
    x, prior,
    2,
    matrix(0, nrow = 3, ncol = 3),
    rep(1, 3),
    c(10, 20, 20)
  )
  expect_equal(result$log_prior_cut_points, -2.944439, tolerance = 1e-5)

  result <- .get_sample_filled(
    x, prior,
    3,
    matrix(0, nrow = 3, ncol = 3),
    rep(1, 3),
    c(6, 13, 20)
  )
  expect_equal(result$log_prior_cut_points, -5.455321, tolerance = 1e-5)
})

test_that('metropolis ratio within', {
  prior2 <- prior
  prior2$t_min <- 7

  base_sample1 <- list(
    n_segments = 1,
    beta = matrix(0, nrow = 3, ncol = 3),
    tau_squared = rep(1, 3),
    cut_points = c(20, 20, 20)
  )
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample1,
      base_sample1,
      x, prior2
    ),
    0
  )
  sample1_2 <- base_sample1
  sample1_2$beta[1, ] <- 0.5
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample1,
      sample1_2,
      x, prior2
    ),
    2.002914, tolerance = 1e-5
  )

  base_sample2 <- list(
    n_segments = 2,
    beta = matrix(0, nrow = 3, ncol = 3),
    tau_squared = rep(1, 3),
    cut_points = c(10, 20, 20)
  )
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      base_sample2,
      x, prior2
    ),
    0
  )

  # Move the cutpoint by a big jump
  sample2 <- base_sample2
  sample2$cut_points[1] <- 5
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      sample2,
      x, prior2
    ),
    0.2749768, tolerance = 1e-5
  )

  # Move the cutpoint by a small jump, unconstrained
  sample2 <- base_sample2
  sample2$cut_points[1] <- 9
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      sample2,
      x, prior2
    ),
    0.1786913, tolerance = 1e-5
  )

  # Move the cutpoint by a small jump, contrained
  sample2 <- base_sample2
  sample2$cut_points[1] <- 7
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      sample2,
      x, prior2
    ),
    0.2207815, tolerance = 1e-5
  )
})
