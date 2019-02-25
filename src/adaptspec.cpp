#include <random>

#include <RcppEigen.h>

#include "adaptspec/sampler.hpp"
#include "adaptspec/samples.hpp"
#include "progress.hpp"
#include "samples.hpp"
#include "utils.hpp"

using namespace bayesspec;

// [[Rcpp::export(name=".adaptspec")]]
Rcpp::List adaptspec(
    unsigned int nLoop,
    unsigned int nWarmUp,
    Rcpp::NumericMatrix xR,
    Rcpp::List missingIndicesR,
    Rcpp::List priorList,
    Rcpp::List tuningList,
    Rcpp::List startR,
    Rcpp::List thin,
    bool showProgress,
    bool debug
) {
    std::mt19937_64 rng(static_cast<uint_fast64_t>(UINT_FAST64_MAX * R::unif_rand()));

    Eigen::MatrixXd x = Rcpp::as<Eigen::MatrixXd>(xR);

    std::vector<Eigen::VectorXi> missingIndices;
    for (Rcpp::IntegerVector missingIndicesI : missingIndicesR) {
        missingIndices.push_back(Rcpp::as<Eigen::VectorXi>(missingIndicesI));
    }

    Rcpp::List xMissingStart = startR["x_missing"];
    for (int i = 0; i < missingIndices.size(); ++i) {
        Rcpp::NumericVector xMissingStartI = xMissingStart[i];
        for (int j = 0; j < missingIndices[i].size(); ++j) {
            x(missingIndices[i][j], i) = xMissingStartI[j];
        }
    }

    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecParameters start = AdaptSpecParameters::fromList(startR, prior);
    AdaptSpecTuning tuning = AdaptSpecTuning::fromList(tuningList);
    AdaptSpecSampler sampler(x, missingIndices, start, tuning, prior);

    AdaptSpecSamples samples(
        nLoop - nWarmUp,
        thin["n_segments"],
        thin["beta"],
        thin["tau_squared"],
        thin["cut_points"],
        thin["mu"],
        prior
    );
    Samples<double> logPosteriorSamples(
        nLoop - nWarmUp,
        thin["log_posterior"]
    );

    std::vector< Samples<double> > xMissingSamples;
    for (int i = 0; i < missingIndices.size(); ++i) {
        xMissingSamples.emplace_back(
            nLoop - nWarmUp,
            thin["x_missing"],
            missingIndices[i].size()
        );
    }
    ProgressBar progressBar(nLoop);
    for (unsigned int iteration = 0; iteration < nLoop; ++iteration) {
        if (iteration == nWarmUp) {
            sampler.endWarmUp();
        }

        if (debug) {
            sampler.debugOutput();
        }

        sampler.sample(rng);

        if (iteration % 100 == 0) {
            // NOTE(mgnb): checks whether the user has pressed Ctrl-C (among other things)
            Rcpp::checkUserInterrupt();
        }

        if (iteration >= nWarmUp) {
            samples.save(sampler.getCurrent());
            logPosteriorSamples.save(sampler.getLogPosterior());
            for (int i = 0; i < missingIndices.size(); ++i) {
                if (missingIndices[i].size() == 0) continue;
                std::vector<double> xMissing(missingIndices[i].size());
                for (int j = 0; j < missingIndices[i].size(); ++j) {
                    xMissing[j] = x(missingIndices[i][j], i);
                }
                xMissingSamples[i].save(xMissing);
            }
        }

        if (showProgress) {
            ++progressBar;
        }
    }

    Rcpp::List xMissingSamplesOutput;
    for (Samples<double> samples : xMissingSamples) {
        xMissingSamplesOutput.push_back(Rcpp::wrap(samples));
    }

    Rcpp::List finalValues = sampler.getCurrent().asList();
    finalValues["x_missing"] = missingValuesAsList(x, missingIndices);

    Rcpp::List output = samples.asList();
    output["log_posterior"] = Rcpp::wrap(logPosteriorSamples);
    output["x_missing"] = xMissingSamplesOutput;
    output["final_values"] = finalValues;
    output["statistics"] = sampler.getStatistics().asList();
    output["warm_up_statistics"] = sampler.getWarmUpStatistics().asList();
    return output;
}

// These functions are used for testing

Rcpp::List wrapState(const AdaptSpecState& state) {
    Rcpp::List output;

    Rcpp::List parameters;
    parameters["n_segments"] = Rcpp::wrap(state.parameters.nSegments);
    parameters["beta"] = Rcpp::wrap(state.parameters.beta);
    parameters["tau_squared"] = Rcpp::wrap(state.parameters.tauSquared);
    parameters["cut_points"] = Rcpp::wrap(state.parameters.cutPoints);
    parameters["mu"] = Rcpp::wrap(state.parameters.mu);
    output["parameters"] = parameters;

    output["segment_lengths"] = Rcpp::wrap(state.segmentLengths);
    Rcpp::List nu;
    Rcpp::List periodogram;
    Rcpp::List precisionCholeskyMode;
    for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
        nu.push_back(Rcpp::wrap(state.nu[segment]));
        periodogram.push_back(Rcpp::wrap(state.periodogram[segment]));
        precisionCholeskyMode.push_back(Rcpp::wrap(state.precisionCholeskyMode[segment]));
    }
    output["nu"] = nu;
    output["periodogram"] = periodogram;
    output["beta_mode"] = Rcpp::wrap(state.betaMode);
    output["precision_cholesky_mode"] = precisionCholeskyMode;
    output["log_segment_proposal"] = Rcpp::wrap(state.logSegmentProposal);
    output["log_segment_likelihood"] = Rcpp::wrap(state.logSegmentLikelihood);
    output["log_segment_prior"] = Rcpp::wrap(state.logSegmentPrior);
    output["log_prior_cut_points"] = Rcpp::wrap(state.logPriorCutPoints);

    return output;
}

AdaptSpecState getStateFromList(
    Rcpp::List parametersList,
    Eigen::MatrixXd& x,
    const AdaptSpecPrior& prior,
    const AdaptSpecTuning& tuning
) {
    return AdaptSpecState(
        AdaptSpecParameters::fromList(parametersList, prior),
        x, prior, tuning
    );
}

// [[Rcpp::export(name=".get_sample_filled")]]
Rcpp::List getSampleFilled(
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    Rcpp::List stateList,
    Rcpp::List tuningList
) {
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecTuning tuning = AdaptSpecTuning::fromList(tuningList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);

    return wrapState(getStateFromList(stateList, x, prior, tuning));
}

// [[Rcpp::export(name=".get_metropolis_log_ratio")]]
double getMetropolisLogRatio(
    Rcpp::List currentR,
    Rcpp::List proposalR,
    Rcpp::NumericMatrix xR,
    Rcpp::List priorList,
    Rcpp::List tuningList
) {
    AdaptSpecPrior prior = AdaptSpecPrior::fromList(priorList);
    AdaptSpecTuning tuning = AdaptSpecTuning::fromList(tuningList);
    Eigen::MatrixXd x = Rcpp::as< Eigen::MatrixXd >(xR);
    AdaptSpecState current = getStateFromList(currentR, x, prior, tuning);
    AdaptSpecState proposal = getStateFromList(proposalR, x, prior, tuning);

    return AdaptSpecState::getMetropolisLogRatio(current, proposal);
}
