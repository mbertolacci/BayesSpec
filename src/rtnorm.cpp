#include <RcppEigen.h>
#include "random/truncated-normal.hpp"

using namespace bayesspec;

// [[Rcpp::export(name="rtnorm")]]
Rcpp::NumericVector rtnorm(
    int n,
    Rcpp::NumericVector mean,
    Rcpp::NumericVector sd,
    Rcpp::NumericVector lower,
    Rcpp::NumericVector upper
) {
    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));
    Rcpp::NumericVector output(n);

    for (int i = 0; i < n; ++i) {
        TruncatedNormalDistribution d(
            mean[i % mean.size()],
            sd[i % sd.size()],
            lower[i % lower.size()],
            upper[i % upper.size()]
        );
        output[i] = d(rng);
    }
    return output;
}
