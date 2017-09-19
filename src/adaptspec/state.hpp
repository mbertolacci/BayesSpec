#ifndef SRC_ADAPTSPEC_STATE_HPP_
#define SRC_ADAPTSPEC_STATE_HPP_

#include <RcppEigen.h>

#include "../random/inverse-gamma.hpp"
#include "../random/truncated-distribution.hpp"
#include "../random/utils.hpp"

#include "../whittle-likelihood.hpp"

#include "parameters.hpp"
#include "prior.hpp"
#include "beta-optimiser.hpp"
#include "utils.hpp"

namespace bayesspec {

inline
unsigned int absDiff(unsigned int a, unsigned int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return b - a;
}

class AdaptSpecState {
public:
    AdaptSpecParameters parameters;

    Eigen::VectorXd segmentLengths;
    std::vector<Eigen::MatrixXd> nu;
    std::vector<Eigen::MatrixXd> periodogram;

    Eigen::MatrixXd betaMle;
    std::vector<Eigen::MatrixXd> precisionCholeskyMle;

    Eigen::VectorXd logSegmentProposal;
    Eigen::VectorXd logSegmentLikelihood;
    Eigen::VectorXd logSegmentPrior;
    double logPriorCutPoints;

    const Eigen::MatrixXd *x;

    AdaptSpecState(
        const AdaptSpecParameters& parameters_,
        const Eigen::MatrixXd& x_,
        const AdaptSpecPrior& prior,
        double probMM1,
        double varInflate
    ) : parameters(parameters_),
        x(&x_),
        prior_(&prior),
        probMM1_(probMM1),
        varInflate_(varInflate) {
        initialise_(false);
    }

    void updateLogPriorCutPoints() {
        logPriorCutPoints = 0;
        for (unsigned int segment = 0; segment < parameters.nSegments - 1; ++segment) {
            logPriorCutPoints -= std::log(
                static_cast<double>(x->rows())
                - (segment == 0 ? 0 : parameters.cutPoints[segment - 1])
                - (static_cast<double>(parameters.nSegments) - static_cast<double>(segment)) * prior_->tMin
                + 1
            );
        }
    }

    void updateSegmentDensities(unsigned int segment) {
        logSegmentProposal[segment] = precisionCholeskyMle[segment].diagonal().array().log().sum()
            - 0.5 * (
                precisionCholeskyMle[segment].template triangularView<Eigen::Upper>() * (
                    parameters.beta.row(segment) - betaMle.row(segment)
                ).transpose()
            ).array().square().sum()
            - 0.5 * (1 + prior_->nBases) * std::log(2 * M_PI);

        logSegmentLikelihood[segment] = logWhittleLikelihood(
            nu[segment] * parameters.beta.row(segment).transpose(),
            periodogram[segment],
            segmentLengths[segment]
        ).sum();

        logSegmentPrior[segment] = (
            -parameters.beta(segment, 0) * parameters.beta(segment, 0) / (2 * prior_->sigmaSquaredAlpha)
            - 0.5 * std::log(prior_->sigmaSquaredAlpha)
            - parameters.beta.row(segment).segment(1, prior_->nBases).array().square().sum() / (2 * parameters.tauSquared[segment])
            - 0.5 * prior_->nBases * std::log(parameters.tauSquared[segment])
            - 0.5 * (1 + prior_->nBases) * std::log(2 * M_PI)
            - std::log(prior_->tauUpperLimit)
        );
    }

    void updateSegmentFit(unsigned int segment) {
        BetaOptimiser optimiser(
            segmentLengths[segment],
            periodogram[segment],
            nu[segment],
            prior_->sigmaSquaredAlpha,
            parameters.tauSquared[segment]
        );

        // Outputs
        Eigen::VectorXd beta(parameters.beta.row(segment).transpose());
        Eigen::VectorXd gradient(prior_->nBases + 1);
        Eigen::MatrixXd hessian(prior_->nBases + 1, prior_->nBases + 1);
        int status = optimiser.run(beta, gradient, hessian);
        if (status != 1) {
            Rcpp::Rcout << "Warning: optimiser failed\n" << optimiser << "\n";
            Rcpp::Rcout << "Current state =\n" << *this << "\n";
            Rcpp::stop("Optimiser failed");
        }

        betaMle.row(segment) = beta.transpose();

        hessian /= varInflate_;
        precisionCholeskyMle[segment] = hessian.llt().matrixU();

        updateSegmentDensities(segment);
    }

    void updateSegment(unsigned int segment) {
        unsigned int segmentLength = segmentLengths[segment];
        nu[segment] = AdaptSpecUtils::calculateNu(segmentLength, prior_->nBases);
        periodogram[segment] = AdaptSpecUtils::calculatePeriodogram(
            *x,
            parameters.cutPoints[segment],
            segmentLength
        );

        updateSegmentFit(segment);
    }

    template<typename RNG>
    void sample(RNG& rng) {
        sampleBetween_(rng);
        sampleWithin_(rng);
        sampleTauSquared_(rng);
    }

    double getLogPosterior() const {
        return logSegmentLikelihood.segment(0, parameters.nSegments).sum() + logSegmentPrior.segment(0, parameters.nSegments).sum() + logPriorCutPoints;
    }

    double getLogSegmentProposal() const {
        return logSegmentProposal.segment(0, parameters.nSegments).sum();
    }

    static double getMetropolisLogRatio(const AdaptSpecState& current, const AdaptSpecState& proposal) {
        if (current.parameters.nSegments == proposal.parameters.nSegments) {
            return getMetropolisLogRatioWithin_(current, proposal);
        } else if (current.parameters.nSegments > proposal.parameters.nSegments) {
            return getMetropolisLogRatioDeath_(current, proposal);
        }
        return getMetropolisLogRatioBirth_(current, proposal);
    }

    friend std::ostream& operator<< (std::ostream& stream, const AdaptSpecState& state) {
        stream << "nSegments = " << state.parameters.nSegments << "\n";
        stream << "cutPoints = " << state.parameters.cutPoints.transpose() << "\n";
        stream << "beta = " << state.parameters.beta.topRows(state.parameters.nSegments) << "\n";
        stream << "tauSquared = " << state.parameters.tauSquared.segment(0, state.parameters.nSegments).transpose() << "\n";
        stream << "segmentLengths = " << state.segmentLengths.transpose() << "\n";
        stream << "logSegmentProposal = " << state.logSegmentProposal.segment(0, state.parameters.nSegments).transpose() << "\n";
        stream << "logSegmentLikelihood = " << state.logSegmentLikelihood.segment(0, state.parameters.nSegments).transpose() << "\n";
        stream << "logSegmentPrior = " << state.logSegmentPrior.segment(0, state.parameters.nSegments).transpose() << "\n";
        stream << "logPriorCutPoints = " << state.logPriorCutPoints << "\n";
        stream << "betaMle = " << state.betaMle.topRows(state.parameters.nSegments) << "\n";
        stream << "xSum = " << state.x->sum() << "\n";
        return stream;
    }

private:
    const AdaptSpecPrior *prior_;
    double probMM1_;
    double varInflate_;

    void initialise_(bool initialiseBetaToMle = false) {
        nu.resize(prior_->nSegmentsMax);
        periodogram.resize(prior_->nSegmentsMax);

        betaMle.resize(prior_->nSegmentsMax, 1 + prior_->nBases);
        betaMle.fill(0);
        precisionCholeskyMle.resize(prior_->nSegmentsMax);
        for (unsigned int segment = 0; segment < prior_->nSegmentsMax; ++segment) {
            precisionCholeskyMle[segment].fill(0);
        }

        logSegmentProposal.resize(prior_->nSegmentsMax);
        logSegmentProposal.fill(0);
        logSegmentLikelihood.resize(prior_->nSegmentsMax);
        logSegmentLikelihood.fill(0);
        logSegmentPrior.resize(prior_->nSegmentsMax);
        logSegmentPrior.fill(0);

        segmentLengths.resize(prior_->nSegmentsMax);
        segmentLengths.fill(0);

        unsigned int lastCutPoint = 0;
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            segmentLengths[segment] = parameters.cutPoints[segment] - lastCutPoint;
            updateSegment(segment);
            if (initialiseBetaToMle) {
                // Initialise the parameters to the MLE
                parameters.beta.row(segment) = betaMle.row(segment);
                updateSegmentDensities(segment);
            }
            lastCutPoint = parameters.cutPoints[segment];
        }
        if (parameters.nSegments > 0) {
            updateLogPriorCutPoints();
        }
    }

    std::vector<unsigned int> getEligibleCuts_() const {
        std::vector<unsigned int> eligibleForCut;
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            if (segmentLengths[segment] > 2 * prior_->tMin) {
                eligibleForCut.push_back(segment);
            }
        }
        return eligibleForCut;
    }

    std::vector<unsigned int> getEligibleMoves_() const {
        if (prior_->nSegmentsMax == prior_->nSegmentsMin) {
            return {};
        }

        bool anyEligibleForCut = (segmentLengths.array() > 2 * prior_->tMin).any();
        if (anyEligibleForCut == 0) {
            if (parameters.nSegments == prior_->nSegmentsMin) {
                return {};
            }
            return { parameters.nSegments - 1 };
        } else {
            if (parameters.nSegments == prior_->nSegmentsMin) {
                return { prior_->nSegmentsMin + 1 };
            } else if (parameters.nSegments == prior_->nSegmentsMax) {
                return { parameters.nSegments - 1 };
            } else {
                return { parameters.nSegments - 1, parameters.nSegments + 1 };
            }
        }
    }

    template<typename RNG>
    void sampleBetaProposal_(unsigned int segment, RNG& rng) {
        Eigen::VectorXd unitNormals(betaMle.cols());
        std::normal_distribution<double> distribution;
        for (unsigned int i = 0; i < unitNormals.size(); ++i) {
            unitNormals[i] = distribution(rng);
        }
        parameters.beta.row(segment) = betaMle.row(segment)
            + precisionCholeskyMle[segment].template triangularView<Eigen::Upper>().solve(
                unitNormals
            ).transpose();

        updateSegmentDensities(segment);
    }

    void moveCutpoint_(unsigned int segment, unsigned int newCutPoint) {
        parameters.cutPoints[segment] = newCutPoint;
        segmentLengths[segment] = segment == 0 ? newCutPoint : newCutPoint - parameters.cutPoints[segment - 1];
        segmentLengths[segment + 1] = parameters.cutPoints[segment + 1] - newCutPoint;

        updateLogPriorCutPoints();
        updateSegment(segment);
        updateSegment(segment + 1);
    }

    template<typename RNG>
    void insertCutPoint_(unsigned int segmentToCut, unsigned int newCutPoint, RNG& rng) {
        ++parameters.nSegments;
        // Shift everything after the cut segment right
        for (unsigned int segment = parameters.nSegments - 1; segment > segmentToCut + 1; --segment) {
            parameters.beta.row(segment) = parameters.beta.row(segment - 1);
            parameters.tauSquared[segment] = parameters.tauSquared[segment - 1];
            parameters.cutPoints[segment] = parameters.cutPoints[segment - 1];
            segmentLengths[segment] = segmentLengths[segment - 1];
            nu[segment] = nu[segment - 1];
            periodogram[segment] = periodogram[segment - 1];
            betaMle.row(segment) = betaMle.row(segment - 1);
            precisionCholeskyMle[segment] = precisionCholeskyMle[segment - 1];
            logSegmentProposal[segment] = logSegmentProposal[segment - 1];
            logSegmentLikelihood[segment] = logSegmentLikelihood[segment - 1];
            logSegmentPrior[segment] = logSegmentPrior[segment - 1];
        }

        parameters.cutPoints[segmentToCut + 1] = parameters.cutPoints[segmentToCut];
        parameters.cutPoints[segmentToCut] = newCutPoint;
        segmentLengths[segmentToCut + 1] = parameters.cutPoints[segmentToCut + 1] - parameters.cutPoints[segmentToCut];
        segmentLengths[segmentToCut] -= segmentLengths[segmentToCut + 1];

        // Draw a new tau-squared
        double oldTauSquared = parameters.tauSquared[segmentToCut];
        double zLower = oldTauSquared / (oldTauSquared + prior_->tauUpperLimit);
        double zUpper = prior_->tauUpperLimit / (oldTauSquared + prior_->tauUpperLimit);
        double z = zLower + randUniform(rng) * (zUpper - zLower);
        parameters.tauSquared[segmentToCut + 1] = oldTauSquared * (1 - z) / z;
        parameters.tauSquared[segmentToCut] = oldTauSquared * z / (1 - z);

        updateLogPriorCutPoints();

        updateSegment(segmentToCut);
        updateSegment(segmentToCut + 1);

        sampleBetaProposal_(segmentToCut, rng);
        sampleBetaProposal_(segmentToCut + 1, rng);
    }

    template<typename RNG>
    void removeCutPoint_(unsigned int segmentToRemove, RNG& rng) {
        --parameters.nSegments;

        double oldTauSquaredLeft = parameters.tauSquared[segmentToRemove];
        double oldTauSquaredRight = parameters.tauSquared[segmentToRemove + 1];

        parameters.cutPoints[segmentToRemove] = parameters.cutPoints[segmentToRemove + 1];
        segmentLengths[segmentToRemove] = segmentLengths[segmentToRemove] + segmentLengths[segmentToRemove + 1];
        parameters.tauSquared[segmentToRemove] = std::sqrt(oldTauSquaredLeft * oldTauSquaredRight);

        // Shift everything after the cut point left
        for (unsigned int segment = segmentToRemove + 1; segment < parameters.nSegments; ++segment) {
            parameters.beta.row(segment) = parameters.beta.row(segment + 1);
            parameters.tauSquared[segment] = parameters.tauSquared[segment + 1];
            parameters.cutPoints[segment] = parameters.cutPoints[segment + 1];
            segmentLengths[segment] = segmentLengths[segment + 1];
            nu[segment] = nu[segment + 1];
            periodogram[segment] = periodogram[segment + 1];
            betaMle.row(segment) = betaMle.row(segment + 1);
            precisionCholeskyMle[segment] = precisionCholeskyMle[segment + 1];
            logSegmentProposal[segment] = logSegmentProposal[segment + 1];
            logSegmentLikelihood[segment] = logSegmentLikelihood[segment + 1];
            logSegmentPrior[segment] = logSegmentPrior[segment + 1];
        }
        segmentLengths[parameters.nSegments] = 0;
        parameters.tauSquared[parameters.nSegments] = 0;
        parameters.beta.row(parameters.nSegments).fill(0);

        updateLogPriorCutPoints();
        updateSegment(segmentToRemove);
        sampleBetaProposal_(segmentToRemove, rng);
    }

    template<typename RNG>
    void sampleBetween_(RNG& rng) {
        std::vector<unsigned int> eligibleMoves = getEligibleMoves_();
        if (eligibleMoves.size() == 0) {
            return;
        }

        unsigned int nSegmentsProposal = eligibleMoves[
            randInteger(0, eligibleMoves.size() - 1, rng)
        ];

        if (nSegmentsProposal == parameters.nSegments) {
            // Do nothing
            return;
        }

        AdaptSpecState proposal(*this);
        if (nSegmentsProposal > parameters.nSegments) {
            std::vector<unsigned int> eligibleForCut = getEligibleCuts_();
            if (eligibleForCut.size() == 0) {
                // Can't cut, so do nothing
                return;
            }
            unsigned int segmentToCut = eligibleForCut[
                randInteger(0, eligibleForCut.size() - 1, rng)
            ];
            unsigned int nPossibleCuts = segmentLengths[segmentToCut] - 2 * prior_->tMin + 1;
            unsigned int newCutPoint = parameters.cutPoints[segmentToCut] - prior_->tMin - randInteger(
                0, nPossibleCuts - 1, rng
            );
            proposal.insertCutPoint_(segmentToCut, newCutPoint, rng);
        } else {
            unsigned int segmentToRemove = randInteger(0, parameters.nSegments - 2, rng);
            proposal.removeCutPoint_(segmentToRemove, rng);
        }

        double alpha = std::min(static_cast<double>(1.0), std::exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (randUniform(rng) < alpha) {
            *this = proposal;
        }
    }

    template<typename RNG>
    void sampleWithin_(RNG& rng) {
        AdaptSpecState proposal(*this);

        if (parameters.nSegments == 1) {
            // Just update the parameters
            proposal.sampleBetaProposal_(0, rng);
        } else {
            // Pick a cutpoint to relocate
            unsigned int segment = randInteger(0, parameters.nSegments - 2, rng);
            unsigned int nMoves = segmentLengths[segment] + segmentLengths[segment + 1] - 2 * prior_->tMin + 1;

            if (nMoves > 1) {
                unsigned int newCutPoint;
                if (randUniform(rng) < probMM1_) {
                    // Make a small move
                    if (segmentLengths[segment] == prior_->tMin) {
                        // The only way is up (baby)
                        newCutPoint = parameters.cutPoints[segment] + randInteger(0, 1, rng);
                    } else if (segmentLengths[segment + 1] == prior_->tMin) {
                        // The only way is down (sadly, no longer a song lyric)
                        newCutPoint = parameters.cutPoints[segment] - randInteger(0, 1, rng);
                    } else {
                        // Go either way
                        newCutPoint = parameters.cutPoints[segment] + randInteger(-1, 1, rng);
                    }
                } else {
                    // Make a big move
                    newCutPoint = parameters.cutPoints[segment + 1] - prior_->tMin - randInteger(0, nMoves - 1, rng);
                }

                proposal.moveCutpoint_(segment, newCutPoint);
            }

            proposal.sampleBetaProposal_(segment, rng);
            proposal.sampleBetaProposal_(segment + 1, rng);
        }

        double alpha = std::min(static_cast<double>(1.0), std::exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (randUniform(rng) < alpha) {
            *this = proposal;
        }
    }

    template<typename RNG>
    void sampleTauSquared_(RNG& rng) {
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            double alpha = static_cast<double>(prior_->nBases) / 2.0 + prior_->tauPriorA;
            double beta = parameters.beta.row(segment).segment(1, prior_->nBases).array().square().sum() / 2.0 + prior_->tauPriorB;
            RightTruncatedDistribution<InverseGammaDistribution> distribution(
                InverseGammaDistribution(alpha, beta),
                prior_->tauUpperLimit
            );
            parameters.tauSquared[segment] = distribution(rng);
            updateSegmentFit(segment);
        }
    }

    // Static methods
    static unsigned int findChangedCutPoint_(const AdaptSpecState& left, const AdaptSpecState& right) {
        unsigned int nMinSegments = std::min(left.parameters.nSegments, right.parameters.nSegments);
        unsigned int changedSegment = nMinSegments;
        for (unsigned int segment = 0; segment < nMinSegments; ++segment) {
            if (left.parameters.cutPoints[segment] != right.parameters.cutPoints[segment]) {
                changedSegment = segment;
                break;
            }
        }
        return changedSegment;
    }

    static double getMetropolisLogRatioBirth_(const AdaptSpecState& current, const AdaptSpecState& proposal) {
        unsigned int segment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = -std::log(static_cast<double>(proposal.getEligibleMoves_().size()));
        double logMoveProposal = -std::log(static_cast<double>(current.getEligibleMoves_().size()));

        double logSegmentChoiceCurrent = -std::log(static_cast<double>(proposal.parameters.nSegments - 1));
        double logSegmentChoiceProposal = -std::log(static_cast<double>(current.getEligibleCuts_().size()));

        unsigned int nPossibleCuts = current.segmentLengths[segment] - 2 * current.prior_->tMin + 1;
        double logCutProposal = -std::log(static_cast<double>(nPossibleCuts));

        double currentTauSquared = current.parameters.tauSquared[segment];
        double tauUpperLimit = current.prior_->tauUpperLimit;
        double zLower = currentTauSquared / (currentTauSquared + tauUpperLimit);
        double zUpper = tauUpperLimit / (currentTauSquared + tauUpperLimit);
        // See insertCutPoint_
        double logZProposal = -std::log(zUpper - zLower);

        double tauSum = std::sqrt(proposal.parameters.tauSquared[segment]) + std::sqrt(proposal.parameters.tauSquared[segment + 1]);
        double logJacobian = std::log(2) + 2 * std::log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            + logSegmentChoiceCurrent - logSegmentChoiceProposal
            - logCutProposal
            - logZProposal
            + logJacobian
        );
    }

    static double getMetropolisLogRatioDeath_(const AdaptSpecState& current, const AdaptSpecState& proposal) {
        unsigned int segment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = -std::log(static_cast<double>(proposal.getEligibleMoves_().size()));
        double logMoveProposal = -std::log(static_cast<double>(current.getEligibleMoves_().size()));

        double logSegmentChoiceCurrent = -std::log(static_cast<double>(proposal.getEligibleCuts_().size()));
        double logSegmentChoiceProposal = -std::log(static_cast<double>(current.parameters.nSegments - 1));

        unsigned int nPossibleCuts = proposal.segmentLengths[segment] - 2 * proposal.prior_->tMin + 1;
        double logCutCurrent = -std::log(static_cast<double>(nPossibleCuts));

        double proposalTauSquared = proposal.parameters.tauSquared[segment];
        double tauUpperLimit = current.prior_->tauUpperLimit;
        double zLower = proposalTauSquared / (proposalTauSquared + tauUpperLimit);
        double zUpper = tauUpperLimit / (proposalTauSquared + tauUpperLimit);
        // See insertCutPoint_
        double logZCurrent = -std::log(zUpper - zLower);

        double tauSum = std::sqrt(current.parameters.tauSquared[segment]) + std::sqrt(current.parameters.tauSquared[segment + 1]);
        double logJacobian = -std::log(2) - 2 * std::log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            + logSegmentChoiceCurrent - logSegmentChoiceProposal
            + logCutCurrent
            + logZCurrent
            + logJacobian
        );
    }

    static double getMetropolisLogRatioWithin_(const AdaptSpecState& current, const AdaptSpecState& proposal) {
        // Within move
        unsigned int nSegments = current.parameters.nSegments;

        // Find the segment that moved, if any
        unsigned int movedSegment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = 0;
        double logMoveProposal = 0;
        if (movedSegment != nSegments
            && current.probMM1_ > 0
            && absDiff(current.parameters.cutPoints[movedSegment], proposal.parameters.cutPoints[movedSegment]) == 1) {
            // Moved only one step, so the jump might not be symmetrical

            logMoveCurrent = -std::log(3);
            logMoveProposal = -std::log(3);
            int tMin = current.prior_->tMin;
            // We know only one or the other can be true, because otherwise
            // nothing would have moved
            if (current.segmentLengths[movedSegment] == tMin || current.segmentLengths[movedSegment + 1] == tMin) {
                logMoveProposal = -std::log(2);
            }
            if (proposal.segmentLengths[movedSegment] == tMin || proposal.segmentLengths[movedSegment + 1] == tMin) {
                logMoveCurrent = -std::log(2);
            }
        }

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
        );
    }
};

}  // namespace bayesspec

#endif  // SRC_ADAPTSPEC_STATE_HPP_
