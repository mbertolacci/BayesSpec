context('adaptspec_stick_breaking_mixture')

test_that('sampler gives sane return samples', {
  x <- cbind(
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20)),
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20))
  )
  design_matrix <- as.matrix(1 : 4)

  result <- adaptspec_stick_breaking_mixture(50, 0, x, design_matrix, 2)
  expect_equal(length(result$components), 2)
  expect_equal(nrow(result$categories), 50)
})
