#include <RcppEigen.h>
#include "random/log-gamma.hpp"

using namespace bayesspec;

// [[Rcpp::export(name="rloggamma")]]
Rcpp::NumericVector rloggamma(int n, Rcpp::NumericVector shape, Rcpp::NumericVector logScale) {
    if (
        (shape.size() == 1 && logScale.size() != 1 && logScale.size() != n) ||
        (logScale.size() == 1 && shape.size() != 1 && shape.size() != n)
    ) {
        Rcpp::stop("Invalid size of input arguments");
    }

    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));
    Rcpp::NumericVector output(n);

    for (int i = 0; i < n; ++i) {
        LogGammaDistribution d(
            shape.size() == 1 ? shape[0] : shape[i],
            logScale.size() == 1 ? logScale[0] : logScale[i]
        );
        output[i] = d(rng);
    }
    return output;
}