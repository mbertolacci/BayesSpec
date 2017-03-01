context('adaptspec')

test_that('sampler gives sane return samples', {
  x <- as.matrix(sin(seq(0, 2 * pi, length.out = 20)))

  result <- adaptspec(50, 0, 4, x, nbasis = 3)
  expect_equal(class(result$n_segments), 'mcmc')
  expect_equal(length(result$n_segments), 50)

  # This one is not a coda::mcmc object
  expect_equal(dim(result$beta), c(50, 4, 4))

  expect_equal(class(result$tau_squared), 'mcmc')
  expect_equal(dim(result$tau_squared), c(50, 4))

  expect_equal(class(result$cut_point), 'mcmc')
  expect_equal(dim(result$tau_squared), c(50, 4))
})
