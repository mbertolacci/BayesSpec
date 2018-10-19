context('adaptspec')

test_that('sampler gives sane return samples', {
  x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

  result <- adaptspec(
    50, 0, x, n_segments_max = 4, n_bases = 3, t_min = 4,
    run_diagnostics = FALSE
  )
  expect_equal(class(result$n_segments), 'mcmc')
  expect_equal(length(result$n_segments), 50)

  expect_equal(class(result$beta), 'mcmca')
  expect_equal(dim(result$beta), c(50, 4, 4))

  expect_equal(class(result$tau_squared), 'mcmc')
  expect_equal(dim(result$tau_squared), c(50, 4))

  expect_equal(class(result$cut_points), 'mcmc')
  expect_equal(dim(result$tau_squared), c(50, 4))

  expect_true(!is.null(result$statistics))
})

test_that('sampler can be initialised with n_segments_min > 1', {
  x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

  result <- adaptspec(
    50, 0, x, n_segments_min = 2, n_segments_max = 2, n_bases = 3, t_min = 4,
    run_diagnostics = FALSE
  )
  expect_equal(class(result$n_segments), 'mcmc')
})

test_that('frequency_transform accepts cbrt', {
  x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

  result <- adaptspec(
    50, 0, x, n_segments_max = 4, n_bases = 3, t_min = 4,
    frequency_transform = 'cbrt',
    run_diagnostics = FALSE
  )
  expect_equal(class(result$n_segments), 'mcmc')
})
