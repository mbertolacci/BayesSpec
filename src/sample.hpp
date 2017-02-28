#ifndef SRC_SAMPLE_HPP_
#define SRC_SAMPLE_HPP_

#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>

#include "cppoptlib/meta.h"
#include "cppoptlib/solver/newtondescentsolver.h"

#include "prior.hpp"
#include "segment_functor.hpp"
#include "splines.hpp"
#include "whittle_likelihood.hpp"

namespace bayesspec {

inline
unsigned int absDiff(unsigned int a, unsigned int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return b - a;
}

class AdaptSpecParameters {
public:
    unsigned int nSegments;
    Eigen::MatrixXd beta;
    Eigen::VectorXd tauSquared;
    Eigen::VectorXi cutPoints;
};

class AdaptSpecSample : public AdaptSpecParameters {
public:
    Eigen::VectorXd segmentLengths;
    std::vector<Eigen::MatrixXd> nu;
    std::vector<Eigen::VectorXd> periodogram;

    Eigen::MatrixXd betaMle;
    std::vector<Eigen::MatrixXd> precisionCholeskyMle;

    Eigen::VectorXd logSegmentProposal;
    Eigen::VectorXd logSegmentLikelihood;
    Eigen::VectorXd logSegmentPrior;
    double logPriorCutPoints;

    AdaptSpecSample(
        const Eigen::VectorXd& x,
        const AdaptSpecPrior& prior,
        unsigned int nStartingSegments = 1
    ) : logPriorCutPoints(0),
        x_(x),
        prior_(prior) {
        nSegments = nStartingSegments;
        beta.resize(prior_.nSegmentsMax, 1 + prior_.nBases);
        beta.fill(0);

        tauSquared.resize(prior_.nSegmentsMax);
        tauSquared.fill(0);

        nu.resize(prior_.nSegmentsMax);
        periodogram.resize(prior_.nSegmentsMax);

        betaMle.resize(prior_.nSegmentsMax, 1 + prior_.nBases);
        betaMle.fill(0);
        precisionCholeskyMle.resize(prior_.nSegmentsMax);
        for (unsigned int segment = 0; segment < prior_.nSegmentsMax; ++segment) {
            precisionCholeskyMle[segment].fill(0);
        }

        logSegmentProposal.resize(prior_.nSegmentsMax);
        logSegmentProposal.fill(0);
        logSegmentLikelihood.resize(prior_.nSegmentsMax);
        logSegmentLikelihood.fill(0);
        logSegmentPrior.resize(prior_.nSegmentsMax);
        logSegmentPrior.fill(0);

        cutPoints.resize(prior_.nSegmentsMax);
        segmentLengths.resize(prior_.nSegmentsMax);
        cutPoints.fill(x_.size());
        segmentLengths.fill(0);

        unsigned int lastCutPoint = 0;
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            cutPoints[segment] = (segment + 1) * x_.size() / nSegments;
            segmentLengths[segment] = cutPoints[segment] - lastCutPoint;
            tauSquared[segment] = prior_.tauUpperLimit * rng.randu();

            updateSegment(segment);
            sampleBetaProposal(segment);

            lastCutPoint = cutPoints[segment];
        }
        if (nSegments > 0) {
            updateLogPriorCutPoints();
        }
    }

    AdaptSpecSample(const AdaptSpecSample& rhs)
        : AdaptSpecParameters(rhs),
          segmentLengths(rhs.segmentLengths),
          nu(rhs.nu),
          periodogram(rhs.periodogram),
          betaMle(rhs.betaMle),
          precisionCholeskyMle(rhs.precisionCholeskyMle),
          logSegmentProposal(rhs.logSegmentProposal),
          logSegmentLikelihood(rhs.logSegmentLikelihood),
          logSegmentPrior(rhs.logSegmentPrior),
          logPriorCutPoints(rhs.logPriorCutPoints),
          x_(rhs.x_),
          prior_(rhs.prior_) {}

    AdaptSpecSample& operator=(const AdaptSpecSample& rhs) {
        // We assume they have the same parent as it is
        nSegments = rhs.nSegments;
        beta = rhs.beta;
        tauSquared = rhs.tauSquared;
        cutPoints = rhs.cutPoints;
        segmentLengths = rhs.segmentLengths;
        nu = rhs.nu;
        periodogram = rhs.periodogram;
        betaMle = rhs.betaMle;
        precisionCholeskyMle = rhs.precisionCholeskyMle;
        logSegmentProposal = rhs.logSegmentProposal;
        logSegmentLikelihood = rhs.logSegmentLikelihood;
        logSegmentPrior = rhs.logSegmentPrior;
        logPriorCutPoints = rhs.logPriorCutPoints;
        return *this;
    }

    void updateLogPriorCutPoints() {
        logPriorCutPoints = 0;
        for (unsigned int segment = 0; segment < nSegments - 1; ++segment) {
            logPriorCutPoints -= log(
                static_cast<double>(x_.size())
                - (segment == 0 ? 0 : cutPoints[segment - 1])
                - (static_cast<double>(nSegments) - static_cast<double>(segment)) * prior_.tMin
                + 1
            );
        }
    }

    void updateSegmentDensities(unsigned int segment) {
        logSegmentProposal[segment] = precisionCholeskyMle[segment].diagonal().array().log().sum()
            - 0.5 * (
                precisionCholeskyMle[segment].triangularView<Eigen::Upper>() * (
                    beta.row(segment) - betaMle.row(segment)
                ).transpose()
            ).array().square().sum()
            - 0.5 * (1 + prior_.nBases) * log(2 * M_PI);

        logSegmentLikelihood[segment] = logWhittleLikelihood(
            nu[segment] * beta.row(segment).transpose(),
            periodogram[segment],
            segmentLengths[segment]
        );

        logSegmentPrior[segment] = (
            -beta(segment, 0) * beta(segment, 0) / (2 * prior_.sigmaSquaredAlpha)
            - 0.5 * log(prior_.sigmaSquaredAlpha)
            - beta.row(segment).segment(1, prior_.nBases).array().square().sum() / (2 * tauSquared[segment])
            - 0.5 * prior_.nBases * log(tauSquared[segment])
            - 0.5 * (1 + prior_.nBases) * log(2 * M_PI)
            - log(prior_.tauUpperLimit)
        );
    }

    void updateSegmentFit(unsigned int segment) {
        SegmentFunctor functor(
            segmentLengths[segment],
            periodogram[segment],
            nu[segment],
            prior_.sigmaSquaredAlpha,
            tauSquared[segment]
        );

        Eigen::VectorXd thisBetaMle = beta.row(segment).transpose();
        cppoptlib::NewtonDescentSolver<SegmentFunctor> solver;
        solver.minimize(functor, thisBetaMle);
        if (solver.status() != cppoptlib::Status::GradNormTolerance) {
            Rcpp::Rcout << "Warning: solver failed with status: " << solver.status() << "\n";
            Rcpp::Rcout << solver.criteria();
            Rcpp::Rcout << "segment details\nx = " << x_.segment(cutPoints[segment] - segmentLengths[segment], segmentLengths[segment]).transpose() << "\n";
            Rcpp::Rcout << "length = " << segmentLengths[segment] << "\n";
            Rcpp::Rcout << "periodogram = " << periodogram[segment].transpose() << "\n";
            Rcpp::Rcout << "nu = " << nu[segment] << "\n";
            Rcpp::stop("Solver failed");
        }

        betaMle.row(segment) = thisBetaMle.transpose();

        Eigen::MatrixXd hessian;
        functor.hessian(thisBetaMle, hessian);
        precisionCholeskyMle[segment] = hessian.llt().matrixU();

        updateSegmentDensities(segment);
    }

    void updateSegment(unsigned int segment) {
        Eigen::FFT<double> fft;
        Eigen::VectorXcd frequencies;

        unsigned int segmentLength = segmentLengths[segment];

        unsigned int nFrequencies = segmentLength / 2;
        Eigen::VectorXd freqs = Eigen::VectorXd::LinSpaced(nFrequencies + 1, 0, 0.5);
        nu[segment] = splineBasis1d(freqs, prior_.nBases, true);

        Eigen::VectorXd thisX = x_.segment(cutPoints[segment] - segmentLength, segmentLength);
        fft.fwd(frequencies, thisX);
        periodogram[segment] = frequencies.segment(0, nFrequencies + 1).cwiseAbs2()
            / static_cast<double>(segmentLength);

        updateSegmentFit(segment);
    }

    void sampleBetaProposal(unsigned int segment) {
        beta.row(segment) = betaMle.row(segment)
            + precisionCholeskyMle[segment].triangularView<Eigen::Upper>().solve(
                rng.randn(betaMle.cols())
            ).transpose();

        updateSegmentDensities(segment);
    }

    void moveCutpoint(unsigned int segment, unsigned int newCutPoint) {
        cutPoints[segment] = newCutPoint;
        segmentLengths[segment] = segment == 0 ? newCutPoint : newCutPoint - cutPoints[segment - 1];
        segmentLengths[segment + 1] = cutPoints[segment + 1] - newCutPoint;

        updateSegment(segment);
        updateSegment(segment + 1);
    }

    void insertCutPoint(unsigned int segmentToCut, unsigned int newCutPoint) {
        ++nSegments;
        // Shift everything after the cut segment right
        for (unsigned int segment = nSegments - 1; segment > segmentToCut + 1; --segment) {
            beta.row(segment) = beta.row(segment - 1);
            tauSquared[segment] = tauSquared[segment - 1];
            cutPoints[segment] = cutPoints[segment - 1];
            segmentLengths[segment] = segmentLengths[segment - 1];
            nu[segment] = nu[segment - 1];
            periodogram[segment] = periodogram[segment - 1];
            betaMle.row(segment) = betaMle.row(segment - 1);
            precisionCholeskyMle[segment] = precisionCholeskyMle[segment - 1];
            logSegmentProposal[segment] = logSegmentProposal[segment - 1];
            logSegmentLikelihood[segment] = logSegmentLikelihood[segment - 1];
            logSegmentPrior[segment] = logSegmentPrior[segment - 1];
        }

        cutPoints[segmentToCut + 1] = cutPoints[segmentToCut];
        cutPoints[segmentToCut] = newCutPoint;
        segmentLengths[segmentToCut + 1] = cutPoints[segmentToCut + 1] - cutPoints[segmentToCut];
        segmentLengths[segmentToCut] -= segmentLengths[segmentToCut + 1];

        // Draw a new tau-squared
        double z = rng.randu();
        double oldTauSquared = tauSquared[segmentToCut];
        tauSquared[segmentToCut + 1] = oldTauSquared * (1 - z) / z;
        tauSquared[segmentToCut] = oldTauSquared * z / (1 - z);

        updateLogPriorCutPoints();

        updateSegment(segmentToCut);
        updateSegment(segmentToCut + 1);

        sampleBetaProposal(segmentToCut);
        sampleBetaProposal(segmentToCut + 1);
    }

    void removeCutPoint(unsigned int segmentToRemove) {
        --nSegments;

        double oldTauSquaredLeft = tauSquared[segmentToRemove];
        double oldTauSquaredRight = tauSquared[segmentToRemove + 1];

        cutPoints[segmentToRemove] = cutPoints[segmentToRemove + 1];
        segmentLengths[segmentToRemove] = segmentLengths[segmentToRemove] + segmentLengths[segmentToRemove + 1];
        tauSquared[segmentToRemove] = sqrt(oldTauSquaredLeft + oldTauSquaredRight);

        // Shift everything after the cut point left
        for (unsigned int segment = segmentToRemove + 1; segment < nSegments; ++segment) {
            beta.row(segment) = beta.row(segment + 1);
            tauSquared[segment] = tauSquared[segment + 1];
            cutPoints[segment] = cutPoints[segment + 1];
            segmentLengths[segment] = segmentLengths[segment + 1];
            nu[segment] = nu[segment + 1];
            periodogram[segment] = periodogram[segment + 1];
            betaMle.row(segment) = betaMle.row(segment + 1);
            precisionCholeskyMle[segment] = precisionCholeskyMle[segment + 1];
            logSegmentProposal[segment] = logSegmentProposal[segment + 1];
            logSegmentLikelihood[segment] = logSegmentLikelihood[segment + 1];
            logSegmentPrior[segment] = logSegmentPrior[segment + 1];
        }
        segmentLengths[nSegments] = 0;
        tauSquared[nSegments] = 0;
        beta.row(nSegments).fill(0);

        updateLogPriorCutPoints();
        updateSegment(segmentToRemove);
        sampleBetaProposal(segmentToRemove);
    }

    std::vector<unsigned int> getEligibleCuts() const {
        std::vector<unsigned int> eligibleForCut;
        for (unsigned int segment = 0; segment < nSegments; ++segment) {
            if (segmentLengths[segment] > 2 * prior_.tMin) {
                eligibleForCut.push_back(segment);
            }
        }
        return eligibleForCut;
    }

    std::vector<unsigned int> getEligibleMoves() const {
        bool anyEligibleForCut = (segmentLengths.array() > 2 * prior_.tMin).any();

        if (anyEligibleForCut == 0) {
            if (nSegments == 1) {
                return { 1 };
            }
            return { nSegments - 1 };
        } else {
            if (nSegments == 1) {
                return { 2 };
            } else if (nSegments == prior_.nSegmentsMax) {
                return { nSegments - 1 };
            } else {
                return { nSegments - 1, nSegments + 1 };
            }
        }
    }

    double getLogPosterior() const {
        return logSegmentLikelihood.segment(0, nSegments).sum() + logSegmentPrior.segment(0, nSegments).sum() + logPriorCutPoints;
    }

    double getLogSegmentProposal() const {
        return logSegmentProposal.segment(0, nSegments).sum();
    }

    static double getMetropolisLogRatio(const AdaptSpecSample& current, const AdaptSpecSample& proposal) {
        if (current.nSegments == proposal.nSegments) {
            return getMetropolisLogRatioWithin_(current, proposal);
        } else if (current.nSegments > proposal.nSegments) {
            return getMetropolisLogRatioDeath_(current, proposal);
        }
        return getMetropolisLogRatioBirth_(current, proposal);
    }

private:
    const Eigen::VectorXd& x_;
    const AdaptSpecPrior& prior_;

    static unsigned int findChangedCutPoint_(const AdaptSpecSample& left, const AdaptSpecSample& right) {
        unsigned int nMinSegments = std::min(left.nSegments, right.nSegments);
        unsigned int changedSegment = nMinSegments;
        for (unsigned int segment = 0; segment < nMinSegments; ++segment) {
            if (left.cutPoints[segment] != right.cutPoints[segment]) {
                changedSegment = segment;
                break;
            }
        }
        return changedSegment;
    }

    static double getMetropolisLogRatioBirth_(const AdaptSpecSample& current, const AdaptSpecSample& proposal) {
        unsigned int segment = findChangedCutPoint_(current, proposal);

        unsigned int nPossibleCuts = current.segmentLengths[segment] - 2 * current.prior_.tMin + 1;

        double logMoveCurrent = -log(static_cast<double>(proposal.getEligibleMoves().size()));
        double logMoveProposal = -log(static_cast<double>(current.getEligibleMoves().size()));
        double logSegmentChoiceProposal = -log(static_cast<double>(current.getEligibleCuts().size()));
        double logCutProposal = -log(static_cast<double>(nPossibleCuts));

        double tauSum = sqrt(proposal.tauSquared[segment]) + sqrt(proposal.tauSquared[segment + 1]);
        double logJacobian = log(2) + 2 * log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            - logSegmentChoiceProposal - logCutProposal
            + logJacobian
        );
    }

    static double getMetropolisLogRatioDeath_(const AdaptSpecSample& current, const AdaptSpecSample& proposal) {
        unsigned int segment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = -log(static_cast<double>(proposal.getEligibleMoves().size()));
        double logMoveProposal = -log(static_cast<double>(current.getEligibleMoves().size()));
        double logSegmentChoiceProposal = -log(static_cast<double>(current.nSegments - 1));
        double tauSum = sqrt(current.tauSquared[segment]) + sqrt(current.tauSquared[segment + 1]);
        double logJacobian = -log(2) + 2 * log(tauSum);

        return (
            proposal.getLogPosterior() - current.getLogPosterior()
            + current.getLogSegmentProposal() - proposal.getLogSegmentProposal()
            + logMoveCurrent - logMoveProposal
            - logSegmentChoiceProposal
            + logJacobian
        );
    }

    static double getMetropolisLogRatioWithin_(const AdaptSpecSample& current, const AdaptSpecSample& proposal) {
        // Within move
        unsigned int nSegments = current.nSegments;

        // Find the segment that moved, if any
        unsigned int movedSegment = findChangedCutPoint_(current, proposal);

        double logMoveCurrent = 0;
        double logMoveProposal = 0;
        if (movedSegment != nSegments
            && absDiff(current.cutPoints[movedSegment], proposal.cutPoints[movedSegment]) == 1) {
            // Moved only one step, so the jump might not be symmetrical

            logMoveCurrent = -log(3);
            logMoveProposal = -log(3);
            int tMin = current.prior_.tMin;
            // We know only one or the other can be true, because otherwise
            // nothing would have moved
            if (current.cutPoints[movedSegment] == tMin || current.cutPoints[movedSegment + 1] == tMin) {
                logMoveProposal = -log(2);
            }
            if (proposal.cutPoints[movedSegment] == tMin || proposal.cutPoints[movedSegment + 1] == tMin) {
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

#endif  // SRC_SAMPLE_HPP_
