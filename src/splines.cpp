#include <RcppEigen.h>
#include "splines.hpp"

using namespace bayesspec;

//' @export
// [[Rcpp::export(name="splines_basis1d_demmler_reinsch")]]
Rcpp::NumericMatrix splines_basis1d_demmler_reinsch(Rcpp::NumericVector xR, unsigned int nBases) {
    typedef Eigen::Map<Eigen::VectorXd> MapVXd;

    MapVXd x = Rcpp::as< MapVXd >(xR);
    if (x.size() > 1) {
        double spacing = x[1] - x[0];
        for (int i = 2; i < x.size(); ++i) {
            if (std::abs(x[i] - x[i - 1] - spacing) > std::sqrt(Eigen::NumTraits<double>::epsilon())) {
                Rcpp::stop("Covariate is not evenly spaced");
            }
        }
    }

    return Rcpp::wrap(splineBasis1dDemmlerReinsch(x, nBases));
}
