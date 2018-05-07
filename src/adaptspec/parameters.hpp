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
            cutPoints[segment] = (segment + 1) * nObservations / nSegments;
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
            beta.row(segment) = segmentBeta;

            lastCutPoint = cutPoints[segment];
        }
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_PARAMETERS_HPP_
