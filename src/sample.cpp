#include <RcppEigen.h>

#include "rng.hpp"
#include "sample.hpp"

using namespace bayesspec;

Rcpp::List wrapSample(const AdaptSpecSample& sample) {
    Rcpp::List output;

    output["n_segments"] = Rcpp::wrap(sample.nSegments);
    output["beta"] = Rcpp::wrap(sample.beta);
    output["tau_squared"] = Rcpp::wrap(sample.tauSquared);
    output["cut_points"] = Rcpp::wrap(sample.cutPoints);
    output["segment_lengths"] = Rcpp::wrap(sample.segmentLengths);
    Rcpp::List nu;
    Rcpp::List periodogram;
    Rcpp::List precisionCholeskyMle;
    for (unsigned int segment = 0; segment < sample.nSegments; ++segment) {
        nu.push_back(Rcpp::wrap(sample.nu[segment]));
        periodogram.push_back(Rcpp::wrap(sample.periodogram[segment]));
        precisionCholeskyMle.push_back(Rcpp::wrap(sample.precisionCholeskyMle[segment]));
    }
    output["nu"] = nu;
    output["periodogram"] = periodogram;
    output["beta_mle"] = Rcpp::wrap(sample.betaMle);
    output["precision_cholesky_mle"] = precisionCholeskyMle;
    output["log_segment_proposal"] = Rcpp::wrap(sample.logSegmentProposal);
    output["log_segment_likelihood"] = Rcpp::wrap(sample.logSegmentLikelihood);
    output["log_segment_prior"] = Rcpp::wrap(sample.logSegmentPrior);
    output["log_prior_cut_points"] = Rcpp::wrap(sample.logPriorCutPoints);

    return output;
}

AdaptSpecSample getSampleFromList(
    Rcpp::List sampleList, const Eigen::MatrixXd& x, const AdaptSpecPrior& prior
) {
    AdaptSpecSample sample(x, prior, 0);
    sample.nSegments = sampleList["n_segments"];
    sample.beta = Rcpp::as< Eigen::MatrixXd >(sampleList["beta"]);
    sample.tauSquared = Rcpp::as< Eigen::VectorXd >(sampleList["tau_squared"]);
    sample.cutPoints = Rcpp::as< Eigen::VectorXi >(sampleList["cut_points"]);

    unsigned int lastCutPoint = 0;
    for (unsigned int segment = 0; segment < sample.nSegments; ++segment) {
        sample.segmentLengths[segment] = sample.cutPoints[segment] - lastCutPoint;
        lastCutPoint = sample.cutPoints[segment];
        sample.updateSegment(segment);
    }
    sample.updateLogPriorCutPoints();

    return sample;
}

// [[Rcpp::export(name=".get_sample_default")]]
Rcpp::List getSampleDefault(
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    unsigned int nStartingSegments
) {
    RNG::initialise();

    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecSample sample(Rcpp::as< Eigen::MatrixXd >(xR), prior, nStartingSegments);

    return wrapSample(sample);
}

// [[Rcpp::export(name=".get_sample_filled")]]
Rcpp::List getSampleFilled(
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    unsigned int nSegments,
    Rcpp::NumericMatrix beta,
    Rcpp::NumericVector tauSquared,
    Rcpp::IntegerVector cutPoints
) {
    RNG::initialise();

    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);

    AdaptSpecSample sample(x, prior, 0);
    sample.nSegments = nSegments;
    sample.beta = Rcpp::as< Eigen::MatrixXd >(beta);
    sample.tauSquared = Rcpp::as< Eigen::VectorXd >(tauSquared);
    sample.cutPoints = Rcpp::as< Eigen::VectorXi >(cutPoints);
    unsigned int lastCutPoint = 0;
    for (unsigned int segment = 0; segment < nSegments; ++segment) {
        sample.segmentLengths[segment] = sample.cutPoints[segment] - lastCutPoint;
        lastCutPoint = sample.cutPoints[segment];
        sample.updateSegment(segment);
    }
    sample.updateLogPriorCutPoints();

    return wrapSample(sample);
}

// [[Rcpp::export(name=".get_metropolis_log_ratio")]]
double getMetropolisLogRatio(
    Rcpp::List currentR,
    Rcpp::List proposalR,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList
) {
    RNG::initialise();

    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);
    AdaptSpecSample current = getSampleFromList(currentR, x, prior);
    AdaptSpecSample proposal = getSampleFromList(proposalR, x, prior);

    return AdaptSpecSample::getMetropolisLogRatio(current, proposal);
}
