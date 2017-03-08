#ifndef SRC_ADAPTSPEC_STATE_HPP_
#define SRC_ADAPTSPEC_STATE_HPP_

#include <RcppEigen.h>

#include "../cppoptlib/meta.h"
#include "../cppoptlib/solver/newtondescentsolver.h"

#include "../truncated-inverse-gamma.hpp"
#include "../whittle-likelihood.hpp"

#include "parameters.hpp"
#include "prior.hpp"
#include "segment-functor.hpp"
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
        double probMM1
    ) : parameters(parameters_),
        x(&x_),
        prior_(&prior),
        probMM1_(probMM1) {
        initialise_();
    }

    AdaptSpecState(
        const Eigen::MatrixXd& x_,
        const AdaptSpecPrior& prior,
        double probMM1,
        unsigned int nStartingSegments = 1
    ) : parameters(prior, x_.rows(), nStartingSegments),
        x(&x_),
        prior_(&prior),
        probMM1_(probMM1) {
        parameters.beta.fill(0);
        parameters.tauSquared.fill(1);
        initialise_();
    }

    void updateData(const Eigen::MatrixXd& x_) {
        x = &x_;
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            updateSegment(segment);
        }
    }

    void updateLogPriorCutPoints() {
        logPriorCutPoints = 0;
        for (unsigned int segment = 0; segment < parameters.nSegments - 1; ++segment) {
            logPriorCutPoints -= log(
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
                precisionCholeskyMle[segment].triangularView<Eigen::Upper>() * (
                    parameters.beta.row(segment) - betaMle.row(segment)
                ).transpose()
            ).array().square().sum()
            - 0.5 * (1 + prior_->nBases) * log(2 * M_PI);

        logSegmentLikelihood[segment] = logWhittleLikelihood(
            nu[segment] * parameters.beta.row(segment).transpose(),
            periodogram[segment],
            segmentLengths[segment]
        ).sum();

        logSegmentPrior[segment] = (
            -parameters.beta(segment, 0) * parameters.beta(segment, 0) / (2 * prior_->sigmaSquaredAlpha)
            - 0.5 * log(prior_->sigmaSquaredAlpha)
            - parameters.beta.row(segment).segment(1, prior_->nBases).array().square().sum() / (2 * parameters.tauSquared[segment])
            - 0.5 * prior_->nBases * log(parameters.tauSquared[segment])
            - 0.5 * (1 + prior_->nBases) * log(2 * M_PI)
            - log(prior_->tauUpperLimit)
        );
    }

    void updateSegmentFit(unsigned int segment) {
        SegmentFunctor functor(
            segmentLengths[segment],
            periodogram[segment],
            nu[segment],
            prior_->sigmaSquaredAlpha,
            parameters.tauSquared[segment]
        );

        Eigen::VectorXd thisBetaMle = parameters.beta.row(segment).transpose();
        cppoptlib::NewtonDescentSolver<SegmentFunctor> solver;
        solver.minimize(functor, thisBetaMle);
        if (solver.status() != cppoptlib::Status::GradNormTolerance) {
            Rcpp::Rcout << "Warning: solver failed with status: " << solver.status() << "\n";
            Rcpp::Rcout << solver.criteria();
            Rcpp::stop("Solver failed");
        }

        betaMle.row(segment) = thisBetaMle.transpose();

        Eigen::MatrixXd hessian(prior_->nBases + 1, prior_->nBases + 1);
        functor.hessian(thisBetaMle, hessian);
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

    void sample() {
        sampleBetween_();
        sampleWithin_();
        sampleTauSquared_();
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
        stream << "logSegmentProposal = " << state.logSegmentProposal.transpose() << "\n";
        stream << "logSegmentLikelihood = " << state.logSegmentLikelihood.transpose() << "\n";
        stream << "logSegmentPrior = " << state.logSegmentPrior.transpose() << "\n";
        stream << "xSum = " << state.x->sum() << "\n";
        return stream;
    }

private:
    const AdaptSpecPrior *prior_;
    double probMM1_;

    void initialise_() {
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
            // Initialise the parameters to the MLE
            parameters.beta.row(segment) = betaMle.row(segment);
            updateSegmentDensities(segment);
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
        bool anyEligibleForCut = (segmentLengths.array() > 2 * prior_->tMin).any();

        if (anyEligibleForCut == 0) {
            if (parameters.nSegments == 1) {
                return { 1 };
            }
            return { parameters.nSegments - 1 };
        } else {
            if (parameters.nSegments == 1) {
                return { 2 };
            } else if (parameters.nSegments == prior_->nSegmentsMax) {
                return { parameters.nSegments - 1 };
            } else {
                return { parameters.nSegments - 1, parameters.nSegments + 1 };
            }
        }
    }

    void sampleBetaProposal_(unsigned int segment) {
        parameters.beta.row(segment) = betaMle.row(segment)
            + precisionCholeskyMle[segment].triangularView<Eigen::Upper>().solve(
                rng.randn(betaMle.cols())
            ).transpose();

        updateSegmentDensities(segment);
    }

    void moveCutpoint_(unsigned int segment, unsigned int newCutPoint) {
        parameters.cutPoints[segment] = newCutPoint;
        segmentLengths[segment] = segment == 0 ? newCutPoint : newCutPoint - parameters.cutPoints[segment - 1];
        segmentLengths[segment + 1] = parameters.cutPoints[segment + 1] - newCutPoint;

        updateSegment(segment);
        updateSegment(segment + 1);
    }

    void insertCutPoint_(unsigned int segmentToCut, unsigned int newCutPoint) {
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
        double z = zLower + rng.randu() * (zUpper - zLower);
        parameters.tauSquared[segmentToCut + 1] = oldTauSquared * (1 - z) / z;
        parameters.tauSquared[segmentToCut] = oldTauSquared * z / (1 - z);

        updateLogPriorCutPoints();

        updateSegment(segmentToCut);
        updateSegment(segmentToCut + 1);

        sampleBetaProposal_(segmentToCut);
        sampleBetaProposal_(segmentToCut + 1);
    }

    void removeCutPoint_(unsigned int segmentToRemove) {
        --parameters.nSegments;

        double oldTauSquaredLeft = parameters.tauSquared[segmentToRemove];
        double oldTauSquaredRight = parameters.tauSquared[segmentToRemove + 1];

        parameters.cutPoints[segmentToRemove] = parameters.cutPoints[segmentToRemove + 1];
        segmentLengths[segmentToRemove] = segmentLengths[segmentToRemove] + segmentLengths[segmentToRemove + 1];
        parameters.tauSquared[segmentToRemove] = sqrt(oldTauSquaredLeft + oldTauSquaredRight);

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
        sampleBetaProposal_(segmentToRemove);
    }

    void sampleBetween_() {
        if (prior_->nSegmentsMax == 1) {
            return;
        }

        std::vector<unsigned int> eligibleMoves = getEligibleMoves_();
        unsigned int nSegmentsProposal = eligibleMoves[rng.randint(0, eligibleMoves.size() - 1)];

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
            unsigned int segmentToCut = eligibleForCut[rng.randint(0, eligibleForCut.size() - 1)];
            unsigned int nPossibleCuts = segmentLengths[segmentToCut] - 2 * prior_->tMin + 1;
            unsigned int newCutPoint = parameters.cutPoints[segmentToCut] - prior_->tMin - rng.randint(
                0,
                nPossibleCuts - 1
            );
            proposal.insertCutPoint_(segmentToCut, newCutPoint);
        } else {
            unsigned int segmentToRemove = rng.randint(0, parameters.nSegments - 2);
            proposal.removeCutPoint_(segmentToRemove);
        }

        double alpha = std::min(1.0, exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (rng.randu() < alpha) {
            *this = proposal;
        }
    }

    void sampleWithin_() {
        AdaptSpecState proposal(*this);

        if (parameters.nSegments == 1) {
            // Just update the parameters
            proposal.sampleBetaProposal_(0);
        } else {
            // Pick a cutpoint to relocate
            unsigned int segment = rng.randint(0, parameters.nSegments - 2);
            unsigned int nMoves = segmentLengths[segment] + segmentLengths[segment + 1] - 2 * prior_->tMin + 1;

            if (nMoves > 1) {
                unsigned int newCutPoint;
                if (rng.randu() < probMM1_) {
                    // Make a small move
                    if (segmentLengths[segment] == prior_->tMin) {
                        // The only way is up (baby)
                        newCutPoint = parameters.cutPoints[segment] + rng.randint(0, 1);
                    } else if (segmentLengths[segment + 1] == prior_->tMin) {
                        // The only way is down (sadly, no longer a song lyric)
                        newCutPoint = parameters.cutPoints[segment] - rng.randint(0, 1);
                    } else {
                        // Go either way
                        newCutPoint = parameters.cutPoints[segment] + rng.randint(-1, 1);
                    }
                } else {
                    // Make a big move
                    newCutPoint = parameters.cutPoints[segment + 1] - prior_->tMin - rng.randint(0, nMoves - 1);
                }

                proposal.moveCutpoint_(segment, newCutPoint);
            }

            proposal.sampleBetaProposal_(segment);
            proposal.sampleBetaProposal_(segment + 1);
        }

        double alpha = std::min(1.0, exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (rng.randu() < alpha) {
            *this = proposal;
        }
    }

    void sampleTauSquared_() {
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            double alpha = static_cast<double>(prior_->nBases) / 2.0 + prior_->tauPriorA;
            double beta = parameters.beta.row(segment).segment(1, prior_->nBases).array().square().sum() / 2.0 + prior_->tauPriorB;
            parameters.tauSquared[segment] = rTruncatedInverseGamma(alpha, beta, prior_->tauUpperLimit);
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

        unsigned int nPossibleCuts = current.segmentLengths[segment] - 2 * current.prior_->tMin + 1;

        double logMoveCurrent = -log(static_cast<double>(proposal.getEligibleMoves_().size()));
        double logMoveProposal = -log(static_cast<double>(current.getEligibleMoves_().size()));
        double logSegmentChoiceProposal = -log(static_cast<double>(current.getEligibleCuts_().size()));
        double logCutProposal = -log(static_cast<double>(nPossibleCuts));

        double tauSum = sqrt(proposal.parameters.tauSquared[segment]) + sqrt(proposal.parameters.tauSquared[segment + 1]);
        double logJacobian = log(2) + 2 * log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            - logSegmentChoiceProposal - logCutProposal
            + logJacobian
        );
    }

    static double getMetropolisLogRatioDeath_(const AdaptSpecState& current, const AdaptSpecState& proposal) {
        unsigned int segment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = -log(static_cast<double>(proposal.getEligibleMoves_().size()));
        double logMoveProposal = -log(static_cast<double>(current.getEligibleMoves_().size()));
        double logSegmentChoiceProposal = -log(static_cast<double>(current.parameters.nSegments - 1));
        double tauSum = sqrt(current.parameters.tauSquared[segment]) + sqrt(current.parameters.tauSquared[segment + 1]);
        double logJacobian = -log(2) + 2 * log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            - logSegmentChoiceProposal
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
            && absDiff(current.parameters.cutPoints[movedSegment], proposal.parameters.cutPoints[movedSegment]) == 1) {
            // Moved only one step, so the jump might not be symmetrical

            logMoveCurrent = -log(3);
            logMoveProposal = -log(3);
            int tMin = current.prior_->tMin;
            // We know only one or the other can be true, because otherwise
            // nothing would have moved
            if (current.parameters.cutPoints[movedSegment] == tMin || current.parameters.cutPoints[movedSegment + 1] == tMin) {
                logMoveProposal = -log(2);
            }
            if (proposal.parameters.cutPoints[movedSegment] == tMin || proposal.parameters.cutPoints[movedSegment + 1] == tMin) {
                logMoveCurrent = -log(2);
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
