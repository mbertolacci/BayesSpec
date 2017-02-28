#include <RcppEigen.h>
#include "splines.hpp"

using namespace bayesspec;

// [[Rcpp::export(name="splines_basis1d")]]
Rcpp::NumericMatrix splineBasis1dR(Rcpp::NumericVector xR, unsigned int nBases, bool omitLinear = false) {
    typedef Eigen::Map<Eigen::VectorXd> MapVXd;

    MapVXd x = Rcpp::as< MapVXd >(xR);
    return Rcpp::wrap(splineBasis1d(
        x, nBases, omitLinear
    ));
}

