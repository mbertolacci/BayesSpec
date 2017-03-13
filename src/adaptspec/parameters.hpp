#ifndef SRC_ADAPTSPEC_PARAMETERS_HPP_
#define SRC_ADAPTSPEC_PARAMETERS_HPP_

#include <RcppEigen.h>

#include "../random/utils.hpp"
#include "../whittle-likelihood.hpp"

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
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_PARAMETERS_HPP_
