#include <RcppEigen.h>
#include "random/log-beta.hpp"

using namespace bayesspec;

// [[Rcpp::export(name="rlogbeta")]]
Rcpp::NumericVector rlogbeta(int n, Rcpp::NumericVector a, Rcpp::NumericVector b) {
    if (
        (a.size() == 1 && b.size() != 1 && b.size() != n) ||
        (b.size() == 1 && a.size() != 1 && a.size() != n)
    ) {
        Rcpp::stop("Invalid size of input arguments");
    }

    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));
    Rcpp::NumericVector output(n);

    for (int i = 0; i < n; ++i) {
        LogBetaDistribution d(
            a.size() == 1 ? a[0] : a[i],
            b.size() == 1 ? b[0] : b[i]
        );
        output[i] = d(rng);
    }
    return output;
}
