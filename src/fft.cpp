#include "fft.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".fft_forward_r2c")]]
Rcpp::ComplexVector fftForwardR2C(Rcpp::NumericVector inputR) {
    Eigen::VectorXd input(Rcpp::as<Eigen::VectorXd>(inputR));
    return Rcpp::wrap(fftForward(input));
}

// [[Rcpp::export(name=".fft_forward_c2c")]]
Rcpp::ComplexVector fftForwardC2C(Rcpp::ComplexVector inputR) {
    Eigen::VectorXcd input(Rcpp::as<Eigen::VectorXcd>(inputR));
    return Rcpp::wrap(fftForward(input));
}

