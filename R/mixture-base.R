.mixture_component_priors <- function(component_model, n_components) {
  if (class(component_model) == 'adaptspecmodel') {
    rep(list(component_model), n_components)
  } else {
    component_model
  }
}

.validate_mixture_component_priors <- function(component_priors, n_components, data) {
  stopifnot(length(component_priors) == n_components)
  for (component_prior in component_priors) {
    # Cannot allow too many segments
    stopifnot(
      nrow(data) >= (component_prior$n_segments_max * component_prior$t_min)
    )
  }
  NULL
}

.mixture_start <- function(start, component_priors, data) {
  missing_indices <- .missing_indices(data)
  n_components <- length(component_priors)
  if (is.null(start$categories)) {
    start$categories <- sample.int(n_components, ncol(data), replace = TRUE) - 1
  }
  if (is.null(start$components)) {
    start$components <- lapply(component_priors, function(component_prior) {
      .adaptspec_start(NULL, component_prior, data)
    })
  }
  start <- .x_missing_start(start, missing_indices)

  start
}

.validate_mixture_start <- function(start, n_components, data) {
  stopifnot(length(start$categories) == ncol(data))
  stopifnot(length(start$components) == n_components)
  missing_indices <- .missing_indices(data)
  for (i in seq_len(ncol(data))) {
    stopifnot(length(missing_indices[[i]]) == length(start$x_missing[[i]]))
  }
}
