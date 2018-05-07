#include <RcppEigen.h>
#include "random/utils.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".rand_gamma")]]
Rcpp::NumericVector randGamma(int n, double shape, double scale) {
    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));
    Rcpp::NumericVector output(n);
    for (int i = 0; i < n; ++i) {
        output[i] = randGamma(shape, scale, rng);
    }
    return output;
}
