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

    // Initialise given prior, data, and starting number of segments
    AdaptSpecParameters(
        const AdaptSpecPrior& prior,
        const Eigen::MatrixXd& x,
        unsigned int nStartingSegments
    ) : AdaptSpecParameters(prior, x.rows(), nStartingSegments) {
        // No data, so just take the defaults from an earlier constructor
        if (x.cols() == 0) return;

        // Initialise the beta parameters from data using their conditional
        // modes
        unsigned int nBases = prior.nBases;
        unsigned int lastCutPoint = 0;
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            unsigned int segmentLength = cutPoints[segment] - lastCutPoint;
            Eigen::MatrixXd segmentPeriodogram = AdaptSpecUtils::calculatePeriodogram(
                x,
                cutPoints[segment],
                segmentLength
            );
            Eigen::MatrixXd segmentNu = AdaptSpecUtils::calculateNu(
                segmentLength,
                nBases
            );

            BetaOptimiser optimiser(
                segmentLength,
                segmentPeriodogram,
                segmentNu,
                prior.sigmaSquaredAlpha,
                tauSquared[segment]
            );

            Eigen::VectorXd segmentBeta(beta.row(segment).transpose());
            Eigen::VectorXd segmentGradient(nBases + 1);
            Eigen::MatrixXd segmentHessian(nBases + 1, nBases + 1);
            int status = optimiser.run(
                segmentBeta,
                segmentGradient,
                segmentHessian
            );
            if (status != 1) {
                Rcpp::stop("Optimiser failed");
            }
            beta.row(segment) = segmentBeta.transpose();

            lastCutPoint = cutPoints[segment];
        }
    }

    AdaptSpecParameters(
        const AdaptSpecPrior& prior,
        const Eigen::MatrixXd& x
    ) : AdaptSpecParameters(prior, x, prior.nSegmentsMin) {}

    bool isValid(const AdaptSpecPrior& prior) {
        // Check that nSegments is valid
        if (nSegments > prior.nSegmentsMax) {
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
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_PARAMETERS_HPP_
