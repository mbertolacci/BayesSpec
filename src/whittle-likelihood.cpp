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

// [[Rcpp::export(name=".log_whittle_likelihood_beta")]]
double logWhittleLikelihoodBeta(
    Rcpp::NumericVector betaR,
    Rcpp::NumericMatrix nuR,
    Rcpp::NumericMatrix periodogramR,
    unsigned int n
) {
    Eigen::VectorXd beta = Rcpp::as<Eigen::VectorXd>(betaR);
    Eigen::MatrixXd nu = Rcpp::as<Eigen::MatrixXd>(nuR);
    Eigen::MatrixXd periodogram = Rcpp::as<Eigen::MatrixXd>(periodogramR);

    Eigen::VectorXd fHat = nu * beta;
    Eigen::MatrixXd devExp = (
        periodogram.array().colwise() / fHat.array().exp()
    ).matrix();

    fHat[0] *= 0.5;
    devExp.row(0) = devExp.row(0) * 0.5;
    if (n % 2 == 0) {
        fHat[n / 2] *= 0.5;
        devExp.row(n / 2) = devExp.row(n / 2) * 0.5;
    }

    return (
        -0.5 * periodogram.cols() * static_cast<double>(n) * log(2 * M_PI)
        - periodogram.cols() * fHat.sum()
        - devExp.sum()
    );
}


// [[Rcpp::export(name=".log_whittle_likelihood_beta_deriv")]]
Rcpp::NumericVector logWhittleLikelihoodBetaDeriv(
    Rcpp::NumericVector betaR,
    Rcpp::NumericMatrix nuR,
    Rcpp::NumericMatrix periodogramR,
    unsigned int n
) {
    Eigen::VectorXd beta = Rcpp::as<Eigen::VectorXd>(betaR);
    Eigen::MatrixXd nu = Rcpp::as<Eigen::MatrixXd>(nuR);
    Eigen::MatrixXd periodogram = Rcpp::as<Eigen::MatrixXd>(periodogramR);

    Eigen::VectorXd fHat = nu * beta;
    Eigen::MatrixXd devExp = (
        periodogram.array().colwise() / fHat.array().exp()
    ).matrix();

    devExp.row(0) = devExp.row(0) * 0.5;
    if (n % 2 == 0) {
        devExp.row(n / 2) = devExp.row(n / 2) * 0.5;
    }

    Eigen::MatrixXd oneMDevExp = (1.0 - devExp.array()).matrix();
    oneMDevExp.row(0) = (oneMDevExp.row(0).array() - 0.5).matrix();
    if (n % 2 == 0) {
        oneMDevExp.row(n / 2) = (oneMDevExp.row(n / 2).array() - 0.5).matrix();
    }
    Eigen::VectorXd gradient(beta);
    gradient.fill(0);
    gradient.noalias() -= (oneMDevExp.transpose() * nu).colwise().sum();

    return Rcpp::wrap(gradient);
}

