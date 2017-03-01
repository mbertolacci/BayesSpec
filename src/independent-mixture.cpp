#include <RcppEigen.h>

#include "adaptspec.hpp"
#include "progress.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".independent_mixture")]]
Rcpp::List independentMixture() {
    RNG::initialise();

    return Rcpp::List();
}
