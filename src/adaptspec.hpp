#ifndef SRC_ADAPTSPEC_HPP_
#define SRC_ADAPTSPEC_HPP_

#include <RcppEigen.h>

#include "rng.hpp"
#include "splines.hpp"
#include "prior.hpp"
#include "sample.hpp"
#include "truncated_inverse_gamma.hpp"
#include "whittle_likelihood.hpp"

namespace bayesspec {

class AdaptSpecSampler {
public:
    AdaptSpecSampler(
        const Eigen::VectorXd& x,
        const AdaptSpecSample& start,
        double probMM1,
        const AdaptSpecPrior& prior
    ) : x_(x),
        probMM1_(probMM1),
        prior_(prior),
        current_(start) {}

    void sample() {
        sampleBetween_();
        sampleWithin_();
        sampleTauSquared_();
    }

    const AdaptSpecParameters& getCurrent() {
        return current_;
    }

private:
    const Eigen::VectorXd x_;
    const double probMM1_;

    const AdaptSpecPrior& prior_;

    AdaptSpecSample current_;

    void sampleBetween_() {
        if (prior_.nSegmentsMax == 1) {
            return;
        }

        std::vector<unsigned int> eligibleMoves = current_.getEligibleMoves();
        unsigned int nSegmentsProposal = eligibleMoves[rng.randint(0, eligibleMoves.size() - 1)];

        if (nSegmentsProposal == current_.nSegments) {
            // Do nothing
            return;
        }

        AdaptSpecSample proposal(current_);
        if (nSegmentsProposal > current_.nSegments) {
            std::vector<unsigned int> eligibleForCut = current_.getEligibleCuts();
            unsigned int segmentToCut = eligibleForCut[rng.randint(0, eligibleForCut.size() - 1)];
            unsigned int nPossibleCuts = current_.segmentLengths[segmentToCut] - 2 * prior_.tMin + 1;
            unsigned int newCutPoint = current_.cutPoints[segmentToCut] - prior_.tMin - rng.randint(
                0,
                nPossibleCuts - 1
            );
            proposal.insertCutPoint(segmentToCut, newCutPoint);
        } else {
            unsigned int segmentToRemove = rng.randint(0, current_.nSegments - 2);
            proposal.removeCutPoint(segmentToRemove);
        }

        double alpha = std::min(1.0, exp(AdaptSpecSample::getMetropolisLogRatio(current_, proposal)));
        if (rng.randu() < alpha) {
            current_ = proposal;
        }
    }

    void sampleWithin_() {
        AdaptSpecSample proposal(current_);

        if (current_.nSegments == 1) {
            // Just update the parameters
            proposal.sampleBetaProposal(0);
        } else {
            // Pick a cutpoint to relocate
            unsigned int segment = rng.randint(0, current_.nSegments - 2);
            unsigned int nMoves = current_.segmentLengths[segment] + current_.segmentLengths[segment + 1] - 2 * prior_.tMin + 1;

            if (nMoves > 1) {
                unsigned int newCutPoint;
                if (rng.randu() < probMM1_) {
                    // Make a small move
                    if (current_.segmentLengths[segment] == prior_.tMin) {
                        // The only way is up (baby)
                        newCutPoint = current_.cutPoints[segment] + rng.randint(0, 1);
                    } else if (current_.segmentLengths[segment + 1] == prior_.tMin) {
                        // The only way is down (sadly, no longer a song lyric)
                        newCutPoint = current_.cutPoints[segment] - rng.randint(0, 1);
                    } else {
                        // Go either way
                        newCutPoint = current_.cutPoints[segment] + rng.randint(-1, 1);
                    }
                } else {
                    // Make a big move
                    newCutPoint = current_.cutPoints[segment + 1] - prior_.tMin - rng.randint(0, nMoves - 1);
                }

                proposal.moveCutpoint(segment, newCutPoint);
            }

            proposal.sampleBetaProposal(segment);
            proposal.sampleBetaProposal(segment + 1);
        }

        double alpha = std::min(1.0, exp(AdaptSpecSample::getMetropolisLogRatio(current_, proposal)));
        if (rng.randu() < alpha) {
            current_ = proposal;
        }
    }

    void sampleTauSquared_() {
        for (unsigned int segment = 0; segment < current_.nSegments; ++segment) {
            double alpha = static_cast<double>(prior_.nBases) / 2.0 + prior_.tauPriorA;
            double beta = current_.beta.row(segment).segment(1, prior_.nBases).array().square().sum() / 2.0 + prior_.tauPriorB;
            current_.tauSquared[segment] = rTruncatedInverseGamma(alpha, beta, prior_.tauUpperLimit);
            current_.updateSegmentFit(segment);
        }
    }
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_HPP_
