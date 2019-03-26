context('adaptspec_lsbp_mixture')

test_that('sampler gives sane return samples', {
  x <- cbind(
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20)),
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20))
  )
  design_matrix <- as.matrix(1 : 4)

  result <- adaptspec_lsbp_mixture(
    50, 0, x, design_matrix, 2,
    component_model = adaptspec_model(n_segments_max = 2, t_min = 10),
    run_diagnostics = FALSE
  )
  expect_equal(length(result$components), 2)
  expect_equal(nrow(result$categories), 50)
  expect_true(!is.null(result$component_statistics))
  expect_true(!is.null(result$component_warm_up_statistics))
})

test_that('sampler can take longer design matrix than data width', {
  x <- cbind(
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20)),
    sin(seq(0, 2 * pi, length.out = 20)),
    cos(seq(0, 2 * pi, length.out = 20))
  )
  design_matrix <- as.matrix(1 : 8)

  result <- adaptspec_lsbp_mixture(
    50, 0, x, design_matrix, 2,
    component_model = adaptspec_model(n_segments_max = 2, t_min = 10),
    run_diagnostics = FALSE
  )
  expect_equal(length(result$components), 2)
  expect_equal(ncol(result$categories), 4)
  expect_equal(nrow(result$categories), 50)
})
