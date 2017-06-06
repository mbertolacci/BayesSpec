#include <RcppEigen.h>
#include "splines.hpp"

using namespace bayesspec;

// [[Rcpp::export(name="splines_basis1d")]]
Rcpp::NumericMatrix splines_basis1d(Rcpp::NumericVector xR, unsigned int nBases, bool omitLinear = false) {
    typedef Eigen::Map<Eigen::VectorXd> MapVXd;

    MapVXd x = Rcpp::as< MapVXd >(xR);
    return Rcpp::wrap(splineBasis1d(
        x, nBases, omitLinear
    ));
}

// [[Rcpp::export]]
Rcpp::List splines_thinplate(const Eigen::MatrixXd& designMatrix, unsigned int nBases) {
    Thinplate2Kernel kernel(designMatrix, nBases);
    if (kernel.info() == Eigen::NumericalIssue) {
        Rcpp::stop("Covariance matrix had NaN elements");
    }
    if (kernel.info() == Eigen::NoConvergence) {
        Rcpp::stop("Convergence issue while computing basis vectors");
    }
    Rcpp::List output;
    output["covariance"] = kernel.covariance();
    output["design_matrix"] = kernel.designMatrix();
    output["eigenvalues"] = kernel.eigenvalues();
    return output;
}
