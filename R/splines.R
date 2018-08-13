.basis_expansion_design_matrix <- function(
  covariates, omega, n_covariates, n_dimensions, n_bases, order, omit_intercept
) {
  stopifnot(n_bases > 0)
  stopifnot(n_bases <= n_covariates)

  intercept_offset <- if (omit_intercept) 0 else 1

  # Construct the design matrix
  n_polynomials <- (order - 1) * n_dimensions
  design_matrix <- matrix(
    1,
    nrow = n_covariates,
    ncol = intercept_offset + n_polynomials + n_bases
  )

  # Null space polynomials
  for (k in 1 : (order - 1)) {
    start <- intercept_offset + 1 + (k - 1) * n_dimensions
    end <- intercept_offset + k * n_dimensions
    design_matrix[, start : end] <- covariates ^ k
  }

  if (n_bases == n_covariates) {
    # We want all eigenvectors, just use the built-in method
    decomposition <- eigen(omega, TRUE)
  } else {
    # RSpectra iteratively calculates as many eigenvalues as you want
    decomposition <- RSpectra::eigs_sym(omega, n_bases)
    if (decomposition$nconv < n_bases) {
      stop('Unable to generate all basis vectors')
    }
  }
  # This is equivalent to decomposition$u %*% diag(sqrt(decomposition$d))
  design_matrix[
    ,
    (intercept_offset + 1 + n_polynomials) :
    (intercept_offset + n_polynomials + n_bases)
  ] <- t(
    t(decomposition$vectors) * sqrt(decomposition$values)
  )

  return(design_matrix)
}

# Polynomial splines on [0, 1] (input scales where necessary)
smoothing_spline_basis <- function(
  covariates,
  n_bases = floor(length(covariates) / 10),
  order = 2,
  omit_intercept = FALSE
) {
  stopifnot(order == 1 || order == 2)
  if (!is.vector(covariates)) {
    covariates <- as.vector(covariates)
  }

  if (max(covariates) > 1 || min(covariates) < 0) {
    # Scale covariates to [0, 1]
    covariates <- (covariates - min(covariates)) / diff(range(covariates))
  }

  n_covariates <- length(covariates)

  # Construct matrices to facilitate pairwise operations
  x <- matrix(rep(covariates, n_covariates), nrow = n_covariates)
  t_x <- t(x)

  pair_min <- pmin(x, t_x)
  if (order == 1) {
    omega <- pair_min
  } else if (order == 2) {
    pair_max <- pmax(x, t_x)
    omega <- (pair_min ^ 2) * (pair_max - pair_min / 3) / 2
  }

  .basis_expansion_design_matrix(
    covariates, omega, n_covariates, 1, n_bases, order, omit_intercept
  )
}

#' Thin plate splines
thinplate_spline_basis <- function(
  covariates,
  n_bases = floor(nrow(covariates) / 10),
  order = 2,
  omit_intercept = FALSE
) {
  flog.debug(
    'Computing thinplate spline (order %d) covariance matrix',
    order,
    name = 'BayesSpec.splines'
  )
  omega <- assist::tp(covariates, order = order)
  flog.debug(
    'Finding %d basis vectors for covariance matrix',
    n_bases,
    name = 'BayesSpec.splines'
  )
  .basis_expansion_design_matrix(
    covariates, omega, nrow(covariates), ncol(covariates),
    n_bases, order, omit_intercept
  )
}
