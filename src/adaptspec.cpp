#include <random>

#include <RcppEigen.h>

#include "adaptspec/sampler.hpp"
#include "adaptspec/samples.hpp"
#include "progress.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".adaptspec")]]
Rcpp::List adaptspec(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    double probMM1,
    unsigned int nSegmentsStart = 1,
    bool showProgress = false
) {
    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));

    Eigen::MatrixXd x = Rcpp::as<Eigen::MatrixXd>(xR);
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecParameters start(prior, x.rows(), nSegmentsStart);
    AdaptSpecSampler sampler(x, start, probMM1, prior);

    AdaptSpecSamples samples(nLoop - nWarmUp, prior);
    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        sampler.sample(rng);

        if (iteration % 100 == 0) {
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            Rcpp::checkUserInterrupt();
        }

        if (iteration >= nWarmUp) {
            samples.save(sampler.getCurrent());
        }

        if (showProgress) {
            ++progressBar;
        }
    }

    return samples.asList();
}

// These functions are used for testing

Rcpp::List wrapState(const AdaptSpecState& state) {
    Rcpp::List output;

    Rcpp::List parameters;
    parameters["n_segments"] = Rcpp::wrap(state.parameters.nSegments);
    parameters["beta"] = Rcpp::wrap(state.parameters.beta);
    parameters["tau_squared"] = Rcpp::wrap(state.parameters.tauSquared);
    parameters["cut_points"] = Rcpp::wrap(state.parameters.cutPoints);
    output["parameters"] = parameters;

    output["segment_lengths"] = Rcpp::wrap(state.segmentLengths);
    Rcpp::List nu;
    Rcpp::List periodogram;
    Rcpp::List precisionCholeskyMle;
    for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
        nu.push_back(Rcpp::wrap(state.nu[segment]));
        periodogram.push_back(Rcpp::wrap(state.periodogram[segment]));
        precisionCholeskyMle.push_back(Rcpp::wrap(state.precisionCholeskyMle[segment]));
    }
    output["nu"] = nu;
    output["periodogram"] = periodogram;
    output["beta_mle"] = Rcpp::wrap(state.betaMle);
    output["precision_cholesky_mle"] = precisionCholeskyMle;
    output["log_segment_proposal"] = Rcpp::wrap(state.logSegmentProposal);
    output["log_segment_likelihood"] = Rcpp::wrap(state.logSegmentLikelihood);
    output["log_segment_prior"] = Rcpp::wrap(state.logSegmentPrior);
    output["log_prior_cut_points"] = Rcpp::wrap(state.logPriorCutPoints);

    return output;
}

AdaptSpecState getStateFromList(
    Rcpp::List parametersList, const Eigen::MatrixXd& x, const AdaptSpecPrior& prior
) {
    AdaptSpecParameters parameters(prior);
    parameters.nSegments = parametersList["n_segments"];
    parameters.beta = Rcpp::as< Eigen::MatrixXd >(parametersList["beta"]);
    parameters.tauSquared = Rcpp::as< Eigen::VectorXd >(parametersList["tau_squared"]);
    parameters.cutPoints = Rcpp::as< Eigen::VectorXi >(parametersList["cut_points"]);

    return AdaptSpecState(parameters, x, prior, 0.8);
}

// [[Rcpp::export(name=".get_sample_default")]]
Rcpp::List getSampleDefault(
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    unsigned int nStartingSegments
) {
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecParameters parameters(prior, xR.rows(), nStartingSegments);
    AdaptSpecState state(parameters, Rcpp::as< Eigen::MatrixXd >(xR), prior, 0.8);

    return wrapState(state);
}

// [[Rcpp::export(name=".get_sample_filled")]]
Rcpp::List getSampleFilled(
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    Rcpp::List stateList
) {
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);

    return wrapState(getStateFromList(stateList, x, prior));
}

// [[Rcpp::export(name=".get_metropolis_log_ratio")]]
double getMetropolisLogRatio(
    Rcpp::List currentR,
    Rcpp::List proposalR,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList
) {
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);
    AdaptSpecState current = getStateFromList(currentR, x, prior);
    AdaptSpecState proposal = getStateFromList(proposalR, x, prior);

    return AdaptSpecState::getMetropolisLogRatio(current, proposal);
}
