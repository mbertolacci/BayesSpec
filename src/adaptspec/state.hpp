#ifndef SRC_ADAPTSPEC_STATE_HPP_
#define SRC_ADAPTSPEC_STATE_HPP_

#include <RcppEigen.h>

#include "../random/utils.hpp"

#include "../whittle-likelihood.hpp"
#include "../whittle-missing.hpp"

#include "parameters.hpp"
#include "prior.hpp"
#include "beta-hmc.hpp"
#include "beta-optimiser.hpp"
#include "statistics.hpp"
#include "tuning.hpp"
#include "utils.hpp"

namespace bayesspec {

inline
unsigned int ceilingDivision(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

inline
int indexOf(int needle, const Eigen::VectorXi& haystack) {
    for (int i = 0; i < haystack.size(); ++i) {
        if (haystack[i] == needle) return i;
    }
    return -1;
}

class AdaptSpecState {
public:
    AdaptSpecParameters parameters;

    Eigen::VectorXd segmentLengths;
    std::vector<Eigen::MatrixXd> nu;
    std::vector<Eigen::MatrixXd> periodogram;

    Eigen::MatrixXd betaMode;
    std::vector<Eigen::MatrixXd> precisionCholeskyMode;

    std::vector<bool> missingDistributionsNeedUpdate;
    std::vector< std::vector<WhittleMissingValuesDistribution> > missingDistributions;

    Eigen::VectorXd logSegmentProposal;
    Eigen::VectorXd logSegmentLikelihood;
    Eigen::VectorXd logSegmentPrior;
    double logPriorCutPoints;

    Eigen::MatrixXd *x;
    const std::vector<Eigen::VectorXi> *missingIndices;

    AdaptSpecState(
        const AdaptSpecParameters& parameters_,
        Eigen::MatrixXd& x_,
        const std::vector<Eigen::VectorXi>& missingIndices_,
        const AdaptSpecPrior& prior,
        const AdaptSpecTuning& tuning
    ) : parameters(parameters_),
        x(&x_),
        missingIndices(&missingIndices_),
        prior_(&prior),
        tuning_(tuning),
        warmedUp_(false) {
        initialise_();
    }

    AdaptSpecState(
        const AdaptSpecParameters& parameters_,
        Eigen::MatrixXd& x_,
        const AdaptSpecPrior& prior,
        const AdaptSpecTuning& tuning
    ) : parameters(parameters_),
        x(&x_),
        missingIndices(NULL),
        prior_(&prior),
        tuning_(tuning),
        warmedUp_(false) {
        initialise_();
    }

    void endWarmUp() {
        warmedUp_ = true;
    }

    void updateLogPriorCutPoints() {
        logPriorCutPoints = 0;
        for (unsigned int segment = 0; segment < parameters.nSegments - 1; ++segment) {
            logPriorCutPoints -= std::log(ceilingDivision(
                static_cast<double>(x->rows())
                - (segment == 0 ? 0 : parameters.cutPoints[segment - 1])
                - (static_cast<double>(parameters.nSegments) - static_cast<double>(segment)) * prior_->tMin
                + 1,
                prior_->timeStep
            ));
        }
    }

    void updateSegmentDensities(unsigned int segment) {
        logSegmentProposal[segment] = precisionCholeskyMode[segment].diagonal().array().log().sum()
            - 0.5 * (
                precisionCholeskyMode[segment].template triangularView<Eigen::Upper>() * (
                    parameters.beta.row(segment) - betaMode.row(segment)
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

    void updateMissingValuesDistributions(unsigned int segment) {
        int segmentUpper = parameters.cutPoints[segment];
        int segmentLower = segmentUpper - segmentLengths[segment];

        Eigen::VectorXd segmentSpectrum = (
            nu[segment] * parameters.beta.row(segment).transpose()
        ).array().exp().matrix();
        for (unsigned int series = 0; series < x->cols(); ++series) {
            std::vector<int> segmentMissingIndices;
            for (int i = 0; i < (*missingIndices)[series].size(); ++i) {
                int index = (*missingIndices)[series][i];
                if (index >= segmentLower && index < segmentUpper) {
                    segmentMissingIndices.push_back(index - segmentLower);
                }
            }

            missingDistributions[segment][series].update(
                x->col(series).segment(segmentLower, segmentLengths[segment]),
                segmentMissingIndices,
                segmentSpectrum
            );
        }
    }

    void updateSegmentFit(unsigned int segment) {
        BetaOptimiser optimiser(
            segmentLengths[segment],
            periodogram[segment],
            nu[segment],
            prior_->sigmaSquaredAlpha,
            parameters.tauSquared[segment],
            tuning_.useHessianCurvature
        );

        // Outputs
        Eigen::VectorXd beta(parameters.beta.row(segment).transpose());
        Eigen::VectorXd gradient(prior_->nBases + 1);
        int status = optimiser.run(beta, gradient, precisionCholeskyMode[segment]);
        if (status != 1) {
            // Reattempt optimisation from a zero start
            beta.fill(0);
            status = optimiser.run(beta, gradient, precisionCholeskyMode[segment]);
            if (status != 1) {
                Rcpp::Rcout << "Warning: optimiser failed\n" << optimiser << "\n";
                Rcpp::Rcout << "Current state =\n" << *this << "\n";
                Rcpp::stop("Optimiser failed");
            }
        }

        betaMode.row(segment) = beta.transpose();
        precisionCholeskyMode[segment] /= std::sqrt(
            warmedUp_ ? tuning_.varInflate : tuning_.warmUpVarInflate
        );

        updateSegmentDensities(segment);
    }

    void updateSegment(unsigned int segment) {
        unsigned int segmentLength = segmentLengths[segment];
        nu[segment] = AdaptSpecUtils::calculateNu(
            segmentLength,
            prior_->nBases,
            prior_->frequencyTransform
        );
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
        if (tuning_.useCutpointWithin) {
           sampleCutpointWithin_(rng);
        }
        if (tuning_.useSingleWithin) {
            sampleSingleWithin_(rng);
        }
        if (tuning_.useHmcWithin) {
            sampleHmcWithin_(rng);
        }
        sampleTauSquared_(rng);
        sampleMissing_(rng);
    }

    double getLogPosterior() const {
        return logSegmentLikelihood.segment(0, parameters.nSegments).sum() + logSegmentPrior.segment(0, parameters.nSegments).sum() + logPriorCutPoints;
    }

    double getLogSegmentProposal() const {
        return logSegmentProposal.segment(0, parameters.nSegments).sum();
    }

    const AdaptSpecStatistics& getStatistics() const {
        return statistics_;
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
        stream << "betaMode = " << state.betaMode.topRows(state.parameters.nSegments) << "\n";
        stream << "xSum = " << state.x->sum() << "\n";
        return stream;
    }

private:
    const AdaptSpecPrior *prior_;
    AdaptSpecTuning tuning_;
    bool warmedUp_;
    AdaptSpecStatistics statistics_;

    void checkParameterValidity_() {
        if (!parameters.isValid(*prior_)) {
            Rcpp::Rcout << "Current state:\n" << *this << "\n";
            Rcpp::stop("Parameters are not valid");
        }
    }

    void initialise_() {
        nu.resize(prior_->nSegmentsMax);
        periodogram.resize(prior_->nSegmentsMax);

        betaMode.resize(prior_->nSegmentsMax, 1 + prior_->nBases);
        betaMode.fill(0);
        precisionCholeskyMode.resize(prior_->nSegmentsMax);
        for (unsigned int segment = 0; segment < prior_->nSegmentsMax; ++segment) {
            precisionCholeskyMode[segment].fill(0);
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
            lastCutPoint = parameters.cutPoints[segment];
        }

        missingDistributionsNeedUpdate.resize(prior_->nSegmentsMax);
        std::fill(
            missingDistributionsNeedUpdate.begin(),
            missingDistributionsNeedUpdate.end(),
            true
        );
        missingDistributions.resize(prior_->nSegmentsMax);
        for (unsigned int segment = 0; segment < prior_->nSegmentsMax; ++segment) {
            missingDistributions[segment].resize(x->cols());
        }

        if (parameters.nSegments > 0) {
            checkParameterValidity_();
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

    void setSegmentBeta_(unsigned int segment, const Eigen::VectorXd& betaNew) {
        parameters.beta.row(segment) = betaNew.transpose();
        checkParameterValidity_();
        updateSegmentDensities(segment);
        missingDistributionsNeedUpdate[segment] = true;
    }

    template<typename RNG>
    void sampleBetaProposal_(unsigned int segment, RNG& rng) {
        Eigen::VectorXd unitNormals(betaMode.cols());
        std::normal_distribution<double> distribution;
        for (unsigned int i = 0; i < unitNormals.size(); ++i) {
            unitNormals[i] = distribution(rng);
        }

        Eigen::VectorXd betaNew = betaMode.row(segment)
            + precisionCholeskyMode[segment].template triangularView<Eigen::Upper>().solve(
                unitNormals
            ).transpose();

        setSegmentBeta_(segment, betaNew);
    }

    void moveCutpoint_(unsigned int segment, unsigned int newCutPoint) {
        if (parameters.cutPoints[segment] == newCutPoint) return;

        parameters.cutPoints[segment] = newCutPoint;
        segmentLengths[segment] = segment == 0 ? newCutPoint : newCutPoint - parameters.cutPoints[segment - 1];
        segmentLengths[segment + 1] = parameters.cutPoints[segment + 1] - newCutPoint;

        checkParameterValidity_();
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
            betaMode.row(segment) = betaMode.row(segment - 1);
            precisionCholeskyMode[segment] = precisionCholeskyMode[segment - 1];
            missingDistributionsNeedUpdate[segment] = missingDistributionsNeedUpdate[segment - 1];
            missingDistributions[segment] = missingDistributions[segment - 1];
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

        checkParameterValidity_();
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
            betaMode.row(segment) = betaMode.row(segment + 1);
            precisionCholeskyMode[segment] = precisionCholeskyMode[segment + 1];
            missingDistributionsNeedUpdate[segment] = missingDistributionsNeedUpdate[segment + 1];
            missingDistributions[segment] = missingDistributions[segment + 1];
            logSegmentProposal[segment] = logSegmentProposal[segment + 1];
            logSegmentLikelihood[segment] = logSegmentLikelihood[segment + 1];
            logSegmentPrior[segment] = logSegmentPrior[segment + 1];
        }
        segmentLengths[parameters.nSegments] = 0;
        parameters.tauSquared[parameters.nSegments] = 0;
        parameters.beta.row(parameters.nSegments).fill(0);

        checkParameterValidity_();
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
            unsigned int nPossibleCuts = ceilingDivision(segmentLengths[segmentToCut] - 2 * prior_->tMin + 1, prior_->timeStep);
            unsigned int previousCutPoint = segmentToCut == 0 ? 0 : parameters.cutPoints[segmentToCut - 1];
            unsigned int newCutPoint = previousCutPoint + prior_->tMin + prior_->timeStep * randInteger(
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
            if (warmedUp_) statistics_.acceptBetween();
        } else {
            if (warmedUp_) statistics_.rejectBetween();
        }
    }

    template<typename RNG>
    void sampleCutpointWithin_(RNG& rng) {
        AdaptSpecState proposal(*this);

        if (parameters.nSegments == 1) {
            // Just update the parameters
            proposal.sampleBetaProposal_(0, rng);
        } else {
            // Pick a cutpoint to relocate
            unsigned int segment = randInteger(0, parameters.nSegments - 2, rng);
            unsigned int nMoves = ceilingDivision(
                segmentLengths[segment] + segmentLengths[segment + 1] - 2 * prior_->tMin + 1,
                prior_->timeStep
            );

            if (nMoves > 1) {
                unsigned int newCutPoint;
                if (randUniform(rng) < tuning_.probShortMove) {
                    // Make a small move

                    int stepSize = prior_->timeStep * randElement(
                        tuning_.shortMoves,
                        tuning_.shortMoveWeights,
                        rng
                    );
                    if (-stepSize > parameters.cutPoints[segment]) {
                        // Step would make cut point negative, reject
                        if (warmedUp_) statistics_.rejectCutpointWithin();
                        return;
                    }
                    newCutPoint = parameters.cutPoints[segment] + stepSize;
                    if (
                        (segment == 0 && newCutPoint < prior_->tMin)
                        || (segment > 0 && newCutPoint <= parameters.cutPoints[segment - 1])
                        || (segment > 0 && newCutPoint - parameters.cutPoints[segment - 1] < prior_->tMin)
                    ) {
                        // Segment too short, reject
                        if (warmedUp_) statistics_.rejectCutpointWithin();
                        return;
                    }

                    if (
                        newCutPoint >= parameters.cutPoints[segment + 1]
                        || parameters.cutPoints[segment + 1] - newCutPoint < prior_->tMin
                    ) {
                        // Next segment too short, reject
                        if (warmedUp_) statistics_.rejectCutpointWithin();
                        return;
                    }
                } else {
                    // Make a big move
                    unsigned int previousCutPoint = segment == 0 ? 0 : parameters.cutPoints[segment - 1];
                    newCutPoint = previousCutPoint + prior_->tMin + prior_->timeStep * randInteger(0, nMoves - 1, rng);
                }

                proposal.moveCutpoint_(segment, newCutPoint);
            }

            proposal.sampleBetaProposal_(segment, rng);
            proposal.sampleBetaProposal_(segment + 1, rng);
        }

        double alpha = std::min(static_cast<double>(1.0), std::exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (randUniform(rng) < alpha) {
            *this = proposal;
            if (warmedUp_) statistics_.acceptCutpointWithin();
        } else {
            if (warmedUp_) statistics_.rejectCutpointWithin();
        }
    }

    template<typename RNG>
    void sampleSingleWithin_(RNG& rng) {
        AdaptSpecState proposal(*this);
        unsigned int segment = randInteger(0, parameters.nSegments - 1, rng);

        proposal.sampleBetaProposal_(segment, rng);

        double alpha = std::min(static_cast<double>(1.0), std::exp(AdaptSpecState::getMetropolisLogRatio(*this, proposal)));
        if (randUniform(rng) < alpha) {
            *this = proposal;
            if (warmedUp_) statistics_.acceptSingleWithin();
        } else {
            if (warmedUp_) statistics_.rejectSingleWithin();
        }

    }

    template<typename RNG>
    void sampleHmcWithin_(RNG& rng) {
        unsigned int segment = randInteger(0, parameters.nSegments - 1, rng);

        Eigen::VectorXd betaCurrent = parameters.beta.row(segment).transpose();
        Eigen::VectorXd betaNew = sampleBetaHmc(
            betaCurrent,
            segmentLengths[segment],
            periodogram[segment],
            nu[segment],
            prior_->sigmaSquaredAlpha,
            parameters.tauSquared[segment],
            tuning_.lMin,
            tuning_.lMax,
            tuning_.epsilonMin,
            tuning_.epsilonMax,
            rng
        );

        if (betaCurrent == betaNew) {
            if (warmedUp_) statistics_.rejectHmcWithin();
            return;
        }

        if (warmedUp_) statistics_.acceptHmcWithin();

        setSegmentBeta_(segment, betaNew);
    }

    template<typename RNG>
    void sampleTauSquared_(RNG& rng) {
        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            double alpha = static_cast<double>(prior_->nBases) / 2.0 + prior_->tauPriorA;
            double beta = parameters.beta.row(segment).segment(1, prior_->nBases).array().square().sum() / 2.0 + prior_->tauPriorB;

            // Draw from right-truncated inverse gamma distribution
            double logConst1 = R::pgamma(
                1 / prior_->tauUpperLimit,
                alpha,
                1 / beta,
                // Gets upper tail probability
                0,
                // Gets log probability
                1
            );
            double u = randUniform(rng);
            double logConst2 = std::log(u) + logConst1;
            parameters.tauSquared[segment] = 1 / R::qgamma(
                logConst2,
                alpha,
                1 / beta,
                // Specifies first argument is upper tail probability
                0,
                // Specifies first argument is log probability
                1
            );

            if (parameters.tauSquared[segment] == 0) {
                Rcpp::Rcout << "Sample of tauSquared[" << segment << "] failed\n"
                    << "  alpha = " << alpha << "\n"
                    << "  beta = " << beta << "\n"
                    << "  logConst1 = " << logConst1 << "\n"
                    << "  logConst2 = " << logConst2 << "\n";
                Rcpp::Rcout << "Current state =\n" << *this << "\n";
                Rcpp::stop("Sample of tauSquared failed");
            }

            checkParameterValidity_();
            updateSegmentFit(segment);
        }
    }

    template<typename RNG>
    void sampleMissing_(RNG& rng) {
        if (missingIndices == NULL) return;

        for (unsigned int segment = 0; segment < parameters.nSegments; ++segment) {
            bool didUpdate = false;
            if (missingDistributionsNeedUpdate[segment]) {
                updateMissingValuesDistributions(segment);
                didUpdate = true;
                missingDistributionsNeedUpdate[segment] = false;
            }

            int segmentLower = parameters.cutPoints[segment] - segmentLengths[segment];
            int nMissingTotal = 0;
            for (unsigned int series = 0; series < x->cols(); ++series) {
                const WhittleMissingValuesDistribution& d = missingDistributions[segment][series];
                int nMissing = d.missingIndices().size();
                nMissingTotal += nMissing;
                if (nMissing == 0) continue;

                Eigen::VectorXd missingSample = d(rng);
                for (int i = 0; i < d.missingIndices().size(); ++i) {
                    if (segmentLower + d.missingIndices()[i] >= x->rows()) {
                        Rcpp::Rcout
                            << "segment = " << segment << " "
                            << "didUpdate = " << didUpdate << " "
                            << "series = " << series << " "
                            << "i = " << i << " "
                            << "segmentLower = " << segmentLower << " "
                            << "d.missingIndices()[i] = " << d.missingIndices()[i] << " "
                            << "x->rows() = " << x->rows() << "\n";
                        Rcpp::stop("Invalid value");
                    }
                    (*x)(segmentLower + d.missingIndices()[i], series) = missingSample[i];
                }
            }

            if (nMissingTotal > 0) {
                updateSegment(segment);
            }
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

        unsigned int nPossibleCuts = ceilingDivision(
            current.segmentLengths[segment] - 2 * current.prior_->tMin + 1,
            current.prior_->timeStep
        );
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

        unsigned int nPossibleCuts = ceilingDivision(
            proposal.segmentLengths[segment] - 2 * proposal.prior_->tMin + 1,
            proposal.prior_->timeStep
        );
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

        unsigned int timeStep = current.prior_->timeStep;

        double logMoveCurrent = 0;
        double logMoveProposal = 0;
        if (movedSegment != nSegments && current.tuning_.probShortMove > 0) {
            int stepSize = (
                proposal.parameters.cutPoints[movedSegment]
                - current.parameters.cutPoints[movedSegment]
            ) / timeStep;

            int currentIndex = indexOf(stepSize, current.tuning_.shortMoves);
            int proposalIndex = indexOf(-stepSize, current.tuning_.shortMoves);

            if (currentIndex != -1 && proposalIndex != -1) {
                logMoveCurrent = std::log(
                    current.tuning_.shortMoveWeights[currentIndex]
                );
                logMoveProposal = std::log(
                    current.tuning_.shortMoveWeights[proposalIndex]
                );
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
