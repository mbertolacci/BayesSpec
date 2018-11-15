.mixture_component_priors <- function(component_model, n_components) {
  if (class(component_model) == 'adaptspecmodel') {
    rep(list(component_model), n_components)
  } else {
    component_model
  }
}

.validate_mixture_component_priors <- function(
  component_priors,
  n_components,
  data
) {
  stopifnot(length(component_priors) == n_components)
  for (component_prior in component_priors) {
    # Cannot allow too many segments
    stopifnot(
      nrow(data) >= (component_prior$n_segments_max * component_prior$t_min)
    )
  }
  NULL
}

.mixture_start <- function(
  start,
  component_priors,
  data,
  first_category_fixed,
  component_tuning,
  initialise_categories = TRUE
) {
  missing_indices <- .missing_indices(data)
  n_components <- length(component_priors)
  if (initialise_categories && is.null(start$categories)) {
    start$categories <- sample.int(n_components, ncol(data), replace = TRUE) - 1
    if (first_category_fixed) {
      start$categories[1] <- 0
    }
  }
  start$components <- lapply(
    seq_len(length(component_priors)),
    function(i) {
      .adaptspec_start(
        start$components[[i]],
        component_priors[[i]],
        data[
          ,
          start$categories == i - 1,
          drop = FALSE
        ],
        component_tuning
      )
    }
  )
  start <- .x_missing_start(start, missing_indices)


  start
}

.validate_mixture_start <- function(
  start,
  n_components,
  component_priors,
  data,
  check_categories = TRUE
) {
  if (check_categories) {
    stopifnot(length(start$categories) == ncol(data))
  }
  stopifnot(length(start$components) == n_components)
  for (i in 1 : length(start$components)) {
    .validate_adaptspec_start(
      start$components[[i]],
      component_priors[[i]],
      data
    )
  }
  .validate_x_missing_start(start, .missing_indices(data))
}
