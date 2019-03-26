context('adaptspec internals')

x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

nu_20_2 <- rbind(
  c(1,  4.501582e-01,  0.22507908),
  c(1,  4.281258e-01,  0.18209280),
  c(1,  3.641856e-01,  0.06955326),
  c(1,  2.645963e-01, -0.06955326),
  c(1,  1.391065e-01, -0.18209280),
  c(1,  2.756424e-17, -0.22507908),
  c(1, -1.391065e-01, -0.18209280),
  c(1, -2.645963e-01, -0.06955326),
  c(1, -3.641856e-01,  0.06955326),
  c(1, -4.281258e-01,  0.18209280),
  c(1, -4.501582e-01,  0.22507908)
)

prior <- adaptspec_model(
  n_segments_max = 3,
  t_min = 1,
  sigma_squared_alpha = 1,
  tau_prior_a = 1,
  tau_prior_b = 1,
  tau_upper_limit = 2,
  n_bases = 2
)
tuning <- .adaptspec_tuning(list(use_hessian_curvature = TRUE))

test_that('initialise', {
  result <- .get_sample_filled(x, prior, .adaptspec_start(
    list(n_segments = 1), prior, x, tuning
  ), tuning)
  expect_equal(result$parameters$cut_points, c(20, 20, 20))
  expect_equal(result$segment_lengths, c(20, 0, 0))

  result <- .get_sample_filled(x, prior, .adaptspec_start(
    list(n_segments = 2), prior, x, tuning
  ), tuning)
  expect_equal(result$parameters$cut_points, c(10, 20, 20))
  expect_equal(result$segment_lengths, c(10, 10, 0))

  result <- .get_sample_filled(x, prior, .adaptspec_start(
    list(n_segments = 3), prior, x, tuning
  ), tuning)
  expect_equal(result$parameters$cut_points, c(6, 13, 20))
  expect_equal(result$segment_lengths, c(6, 7, 7))
})

test_that('fit and densities', {
  result <- .get_sample_filled(
    x, prior,
    list(
      n_segments = 1,
      beta = matrix(0, nrow = 3, ncol = 3),
      tau_squared = rep(1, 3),
      cut_points = c(20, 20, 20)
    ),
    tuning
  )

  # Basis
  expect_equal(result$periodogram[[1]], matrix(c(
    0.000000000, 4.699544030, 0.026898256, 0.008475845, 0.004491217,
    0.002946380, 0.002191168, 0.001779383, 0.001548366, 0.001429210,
    0.001392288
  ), ncol = 1))
  expect_equal(result$nu[[1]], nu_20_2, tolerance = 1e-5)

  # Likelihood and prior
  expect_equal(result$log_segment_likelihood[1], -23.12877, tolerance = 1e-5)
  expect_equal(result$log_segment_prior[1], -3.256816, tolerance = 1e-5)

  # Proposal distribution
  expect_equal(result$log_segment_proposal[1], -11.74658, tolerance = 1e-5)
  expect_equal(
    result$beta_mode[1, ],
    c(-2.106650, 3.223895, 1.371528),
    tolerance = 1e-5
  )
  expect_equal(result$precision_cholesky_mode[[1]], rbind(
    c(2.982172, 1.081056, 0.45990905),
    c(0.000000, 1.117357, 0.07977929),
    c(0.000000, 0.000000, 1.01976571)
  ), tolerance = 1e-5)

  # Multivariate fit
  result <- .get_sample_filled(
    cbind(x, x), prior,
    list(
      n_segments = 1,
      beta = matrix(0, nrow = 3, ncol = 3),
      tau_squared = rep(1, 3),
      cut_points = c(20, 20, 20)
    ),
    tuning
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
  expect_equal(result$nu[[1]], nu_20_2, tolerance = 1e-5)

  # Likelihood and prior
  expect_equal(result$log_segment_likelihood[1], -46.25754, tolerance = 1e-5)
  expect_equal(result$log_segment_prior[1], -3.256816, tolerance = 1e-5)

  # Proposal distribution
  expect_equal(result$log_segment_proposal[1], -52.59283, tolerance = 1e-5)
  expect_equal(
    result$beta_mode[1, ],
    c(-3.217252, 5.486818, 2.548590),
    tolerance = 1e-5
  )
  expect_equal(result$precision_cholesky_mode[[1]], rbind(
    c(4.21696, 1.301131, 0.6043667),
    c(0.00000, 1.467915, 0.1732795),
    c(0.00000, 0.000000, 1.0657376)
  ), tolerance = 1e-5)
})

test_that('the nu matrix is correct for even and odd numbers of observations', {
  y_even <- as.matrix(rnorm(4))
  result_even <- .get_sample_filled(y_even, prior, .adaptspec_start(
    list(n_segments = 1), prior, y_even, tuning
  ), tuning)
  expect_equal(result_even$nu[[1]], rbind(
    c(1,  4.501582e-01,  0.2250791),
    c(1,  2.756424e-17, -0.2250791),
    c(1, -4.501582e-01,  0.2250791)
  ), tolerance = 1e-5)

  y_odd <- as.matrix(rnorm(5))
  result_odd <- .get_sample_filled(y_odd, prior, .adaptspec_start(
    list(n_segments = 1), prior, y_odd, tuning
  ), tuning)
  expect_equal(result_odd$nu[[1]], rbind(
    c(1,  0.4501582,  0.22507908),
    c(1,  0.1391065, -0.18209280),
    c(1, -0.3641856,  0.06955326)
  ), tolerance = 1e-5)
})

test_that('log_prior_cut_points', {
  result <- .get_sample_filled(
    x, prior,
    list(
      n_segments = 1,
      beta = matrix(0, nrow = 3, ncol = 3),
      tau_squared = rep(1, 3),
      cut_points = c(20, 20, 20)
    ),
    tuning
  )
  expect_equal(result$log_prior_cut_points, 0)

  result <- .get_sample_filled(
    x, prior,
    list(
      n_segments = 2,
      beta = matrix(0, nrow = 3, ncol = 3),
      tau_squared = rep(1, 3),
      cut_points = c(10, 20, 20)
    ),
    tuning
  )
  expect_equal(result$log_prior_cut_points, -2.944439, tolerance = 1e-5)

  result <- .get_sample_filled(
    x, prior,
    list(
      n_segments = 3,
      beta = matrix(0, nrow = 3, ncol = 3),
      tau_squared = rep(1, 3),
      cut_points = c(6, 13, 20)
    ),
    tuning
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
      x, prior2, tuning
    ),
    0
  )
  sample1_2 <- base_sample1
  sample1_2$beta[1, ] <- 0.5
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample1,
      sample1_2,
      x, prior2, tuning
    ),
    2.077288, tolerance = 1e-5
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
      x, prior2, tuning
    ),
    0
  )

  # Move the cutpoint by a big jump
  sample2 <- base_sample2
  sample2$cut_points[1] <- 7
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      sample2,
      x, prior2, tuning
    ),
    -0.07241276, tolerance = 1e-5
  )

  # Move the cutpoint by a small jump, unconstrained
  sample2 <- base_sample2
  sample2$cut_points[1] <- 9
  expect_equal(
    .get_metropolis_log_ratio(
      base_sample2,
      sample2,
      x, prior2, tuning
    ),
    -0.009447008, tolerance = 1e-5
  )

  # Move the cutpoint down by a small jump, constrained
  sample1 <- base_sample2
  sample1$cut_points[1] <- 8
  sample2 <- base_sample2
  sample2$cut_points[1] <- 7
  expect_equal(
    .get_metropolis_log_ratio(
      sample1,
      sample2,
      x, prior2, tuning
    ),
    -0.03213229, tolerance = 1e-5
  )

  # Move the cutpoint up by a small jump, constrained
  sample1 <- base_sample2
  sample1$cut_points[1] <- 12
  sample2 <- base_sample2
  sample2$cut_points[1] <- 13
  expect_equal(
    .get_metropolis_log_ratio(
      sample1,
      sample2,
      x, prior2, tuning
    ),
    -0.03213229, tolerance = 1e-5
  )
})
