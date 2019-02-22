#include "whittle-missing.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".sample_whittle_missing")]]
Rcpp::NumericVector sampleWhittleMissingR(
    Rcpp::NumericVector xR,
    Rcpp::IntegerVector missingIndicesR,
    Rcpp::NumericVector halfSpectrumR,
    double mu
) {
    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));

    return Rcpp::wrap(WhittleMissingValuesDistribution(
        Rcpp::as<Eigen::VectorXd>(xR),
        Rcpp::as< std::vector<int> >(missingIndicesR),
        Rcpp::as<Eigen::VectorXd>(halfSpectrumR),
        mu
    )(rng));
}
