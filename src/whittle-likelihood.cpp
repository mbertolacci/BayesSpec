#include <RcppEigen.h>
#include "whittle-likelihood.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".log_whittle_likelihood")]]
Rcpp::NumericVector logWhittleLikelihood(
    Rcpp::NumericVector fHatR,
    Rcpp::NumericMatrix periodogramR,
    unsigned int n
) {
    Eigen::VectorXd fHat = Rcpp::as<Eigen::VectorXd>(fHatR);
    Eigen::MatrixXd periodogram = Rcpp::as<Eigen::MatrixXd>(periodogramR);
    return Rcpp::wrap(logWhittleLikelihood(fHat, periodogram, n));
}
