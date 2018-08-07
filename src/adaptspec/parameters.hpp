#ifndef SRC_ADAPTSPEC_PARAMETERS_HPP_
#define SRC_ADAPTSPEC_PARAMETERS_HPP_

#include <RcppEigen.h>

#include "../random/utils.hpp"
#include "../whittle-likelihood.hpp"

#include "beta-optimiser.hpp"
#include "prior.hpp"
#include "utils.hpp"

namespace bayesspec {

class AdaptSpecParameters {
public:
    unsigned int nSegments;
    Eigen::MatrixXd beta;
    Eigen::VectorXd tauSquared;
    Eigen::VectorXi cutPoints;

    AdaptSpecParameters(const AdaptSpecPrior& prior) {
        beta.resize(prior.nSegmentsMax, 1 + prior.nBases);
        tauSquared.resize(prior.nSegmentsMax);
        cutPoints.resize(prior.nSegmentsMax);
    }

    AdaptSpecParameters(
        const AdaptSpecPrior& prior,
        unsigned int nObservations,
        unsigned int nStartingSegments
    ) : AdaptSpecParameters(prior) {
        nSegments = nStartingSegments;
        cutPoints.fill(nObservations);
        beta.fill(0.1);
        tauSquared.fill(1);
        // Split evenly
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            cutPoints[segment] = prior.timeStep * (
                ((segment + 1) * nObservations) / (nSegments * prior.timeStep)
            );
            tauSquared[segment] = prior.tauUpperLimit / 2;
        }
    }

    AdaptSpecParameters(
        const AdaptSpecPrior& prior,
        unsigned int nObservations
    ) : AdaptSpecParameters(prior, nObservations, prior.nSegmentsMin) {}

    bool isValid(const AdaptSpecPrior& prior) {
        // Check that nSegments is valid
        if (nSegments < prior.nSegmentsMin || nSegments > prior.nSegmentsMax) {
            return false;
        }

        // Increasing sequence of cut points
        for (unsigned int segment = 1; segment < prior.nSegmentsMax; ++segment) {
            if (cutPoints[segment - 1] > cutPoints[segment]) {
                return false;
            }
        }
        unsigned int previousCutPoint = 0;
        // Distance between cut points at least tMin
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            if (cutPoints[segment] - previousCutPoint < prior.tMin) {
                return false;
            }
            previousCutPoint = cutPoints[segment];
        }
        // Cut points satisfy timeStep
        for (unsigned int segment = 0; segment < nSegments - 1; ++segment) {
            if (cutPoints[segment] % prior.timeStep != 0) {
                return false;
            }
        }
        // tauSquared satisfies upper limit
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            if (tauSquared[segment] >= prior.tauUpperLimit) {
                return false;
            }
        }
        return true;
    }

    Rcpp::List asList() const {
        Rcpp::List output;
        output["n_segments"] = nSegments;
        output["beta"] = Rcpp::wrap(beta);
        output["cut_points"] = Rcpp::wrap(cutPoints);
        output["tau_squared"] = Rcpp::wrap(tauSquared);
        return output;
    }

    static AdaptSpecParameters fromList(
        const Rcpp::List& startList,
        const AdaptSpecPrior& prior
    ) {
        AdaptSpecParameters start(prior);
        start.nSegments = startList["n_segments"];
        start.beta = Rcpp::as<Eigen::MatrixXd>(startList["beta"]);
        start.cutPoints = Rcpp::as<Eigen::VectorXi>(startList["cut_points"]);
        start.tauSquared = Rcpp::as<Eigen::VectorXd>(startList["tau_squared"]);

        if (!start.isValid(prior)) {
            throw std::runtime_error("Invalid starting values");
        }
        return start;
    }

    static std::vector<AdaptSpecParameters> fromListOfLists(
        const Rcpp::List& startsList,
        const std::vector<AdaptSpecPrior>& priors
    ) {
        std::vector<AdaptSpecParameters> starts;
        for (unsigned int i = 0; i < startsList.size(); ++i) {
            starts.push_back(fromList(startsList[i], priors[i]));
        }
        return starts;
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_PARAMETERS_HPP_
