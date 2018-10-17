#include <RcppEigen.h>
#include "splines.hpp"

using namespace bayesspec;

//' @export
// [[Rcpp::export(name="splines_basis1d_demmler_reinsch")]]
Rcpp::NumericMatrix splines_basis1d_demmler_reinsch(Rcpp::NumericVector xR, unsigned int nBases) {
    typedef Eigen::Map<Eigen::VectorXd> MapVXd;

    MapVXd x = Rcpp::as< MapVXd >(xR);
    return Rcpp::wrap(splineBasis1dDemmlerReinsch(x, nBases));
}
