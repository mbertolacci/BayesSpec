#ifndef SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_
#define SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_

#include <RcppEigen.h>

#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"
#include "../adaptspec/state.hpp"

namespace bayesspec {

class AdaptSpecMixtureComponentState {
public:
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArray;

    std::vector<Eigen::MatrixXd> allPeriodograms;
    Eigen::MatrixXd allLogSegmentLikelihoods;

    AdaptSpecState state;

    Eigen::MatrixXd *x;
    std::vector<Eigen::VectorXi> missingIndices;

    AdaptSpecMixtureComponentState(
        Eigen::MatrixXd& x_,
        const std::vector<Eigen::VectorXi>& missingIndices_,
        const AdaptSpecParameters& start,
        const AdaptSpecPrior& prior,
        const AdaptSpecTuning& componentTuning
    ) : state(start, x_, missingIndices_, prior, componentTuning),
        x(&x_),
        missingIndices(missingIndices_),
        lastIsComponent_(x->cols()),
        isFirstSample_(true),
        nSegmentsMax_(prior.nSegmentsMax) {
        allPeriodograms.resize(prior.nSegmentsMax);
        allLogSegmentLikelihoods.resize(x->cols(), prior.nSegmentsMax);

        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            // We initialised with all the data, so the state has computed all
            // the periodograms for us
            allPeriodograms[segment] = state.periodogram[segment];

            allLogSegmentLikelihoods.col(segment) = logWhittleLikelihood(
                state.nu[segment] * state.parameters.beta.row(segment).transpose(),
                allPeriodograms[segment],
                state.segmentLengths[segment]
            );
        }
    }

    void endWarmUp() {
        state.endWarmUp();
    }

    template<typename RNG>
    void sample(const BoolArray& isComponent, unsigned int count, RNG& rng) {
        bool hasChanged = isFirstSample_ || !(lastIsComponent_ == isComponent).all();
        isFirstSample_ = false;
        lastIsComponent_ = isComponent;

        Eigen::MatrixXd thisX(x->rows(), count);
        std::vector<Eigen::VectorXi> thisMissingIndices;

        if (hasChanged) {
            for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
                state.periodogram[segment].resize(state.periodogram[segment].rows(), count);
            }
            for (unsigned int segment = 0; segment < nSegmentsMax_; ++segment) {
                state.missingDistributions[segment].resize(count);
            }
            // NOTE(mgnb): this implies a bit of extra work than is strictly
            // necessarily, in that probably only one or two of the series have
            // changed. But this is easier.
            std::fill(
                state.missingDistributionsNeedUpdate.begin(),
                state.missingDistributionsNeedUpdate.end(),
                true
            );
        }

        unsigned int currentIndex = 0;
        for (unsigned int series = 0; series < isComponent.size(); ++series) {
            if (!isComponent[series]) continue;

            thisX.col(currentIndex) = x->col(series);
            thisMissingIndices.emplace_back(missingIndices[series]);
            for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
                state.periodogram[segment].col(currentIndex) = allPeriodograms[segment].col(series);
            }
            ++currentIndex;
        }
        state.x = &thisX;
        state.missingIndices = &thisMissingIndices;
        if (hasChanged) {
            updateAllSegments_();
        }

        AdaptSpecParameters oldParameters = state.parameters;
        Eigen::VectorXd oldSegmentLengths = state.segmentLengths;
        state.sample(rng);
        updateInternals_(isComponent, oldParameters, oldSegmentLengths);

        currentIndex = 0;
        for (unsigned int series = 0; series < isComponent.size(); ++series) {
            if (!isComponent[series]) continue;
            for (unsigned int i = 0; i < thisMissingIndices[currentIndex].size(); ++i) {
                (*x)(thisMissingIndices[currentIndex][i], series) = thisX(thisMissingIndices[currentIndex][i], currentIndex);
            }
            ++currentIndex;
        }
    }

    template<typename RNG>
    void proposeSpectra(const BoolArray& isComponent, unsigned int count, RNG& rng) {
        bool hasChanged = !(lastIsComponent_ == isComponent).all();

        Eigen::MatrixXd thisX(x->rows(), count);

        if (hasChanged) {
            for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
                state.periodogram[segment].resize(state.periodogram[segment].rows(), count);
            }
            for (unsigned int segment = 0; segment < nSegmentsMax_; ++segment) {
                state.missingDistributions[segment].resize(count);
            }
            // NOTE(mgnb): this implies a bit of extra work than is strictly
            // necessarily, in that probably only one or two of the series have
            // changed. But this is easier.
            std::fill(
                state.missingDistributionsNeedUpdate.begin(),
                state.missingDistributionsNeedUpdate.end(),
                true
            );
        }

        unsigned int currentIndex = 0;
        for (unsigned int series = 0; series < isComponent.size(); ++series) {
            if (!isComponent[series]) continue;

            thisX.col(currentIndex) = x->col(series);
            for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
                state.periodogram[segment].col(currentIndex) = allPeriodograms[segment].col(series);
            }
            ++currentIndex;
        }
        state.x = &thisX;
        if (hasChanged) {
            updateAllSegments_();
        }

        AdaptSpecParameters oldParameters = state.parameters;
        Eigen::VectorXd oldSegmentLengths = state.segmentLengths;
        state.proposeSpectra(rng);
        updateInternals_(isComponent, oldParameters, oldSegmentLengths);
    }

    double getLogSegmentProposal() const {
        return state.getLogSegmentProposal();
    }

private:
    BoolArray lastIsComponent_;
    bool isFirstSample_;
    unsigned int nSegmentsMax_;

    void updateAllSegments_() {
        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            state.means[segment] = state.x->cols() == 0 ? 0 : state.x->block(
                state.parameters.cutPoints[segment] - state.segmentLengths[segment],
                0,
                state.segmentLengths[segment],
                state.x->cols()
            ).mean();
            state.updateSegmentFit(segment);
        }
    }

    void updateInternals_(
        const Eigen::Array<bool, Eigen::Dynamic, 1>& isComponent,
        const AdaptSpecParameters& oldParameters,
        const Eigen::VectorXd& oldSegmentLengths
    ) {
        std::vector<Eigen::MatrixXd> oldAllPeriodograms = allPeriodograms;
        Eigen::MatrixXd oldAllLogSegmentLikelihoods = allLogSegmentLikelihoods;

        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            bool hasMatchingSegment = false;

            unsigned int matchingOldSegment = 0;
            for (unsigned int segmentOld = 0; segmentOld < oldParameters.nSegments; ++segmentOld) {
                if (state.parameters.cutPoints[segment] == oldParameters.cutPoints[segmentOld]
                    && state.segmentLengths[segment] == oldSegmentLengths[segmentOld]) {
                    // Has the same cutpoint and length
                    hasMatchingSegment = true;
                    matchingOldSegment = segmentOld;
                    break;
                }
            }

            bool mustUpdateLikelihoods = true;
            if (hasMatchingSegment) {
                allPeriodograms[segment] = oldAllPeriodograms[matchingOldSegment];

                if (
                    state.parameters.beta.row(segment) == oldParameters.beta.row(matchingOldSegment)
                    && state.parameters.mu[segment] == oldParameters.mu[segment]
                ) {
                    // Same beta as well, so log likelihood is unchanged
                    allLogSegmentLikelihoods.col(segment) = oldAllLogSegmentLikelihoods.col(matchingOldSegment);
                    mustUpdateLikelihoods = false;
                }
            } else {
                allPeriodograms[segment].resize(state.periodogram[segment].rows(), x->cols());

                unsigned int currentIndex = 0;
                for (unsigned int series = 0; series < x->cols(); ++series) {
                    if (isComponent[series]) {
                        allPeriodograms[segment].col(series) = state.periodogram[segment].col(currentIndex);
                        ++currentIndex;
                    } else {
                        allPeriodograms[segment].col(series) = AdaptSpecUtils::calculatePeriodogram(
                            x->col(series),
                            state.parameters.cutPoints[segment],
                            state.segmentLengths[segment]
                        );
                    }
                }
            }

            if (mustUpdateLikelihoods) {
                AdaptSpecUtils::updatePeriodogramWithMean(
                    allPeriodograms[segment],
                    *x,
                    state.parameters.cutPoints[segment],
                    state.segmentLengths[segment],
                    state.parameters.mu[segment]
                );
                allLogSegmentLikelihoods.col(segment) = logWhittleLikelihood(
                    state.nu[segment] * state.parameters.beta.row(segment).transpose(),
                    allPeriodograms[segment],
                    state.segmentLengths[segment]
                );
            }
        }
    }
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_
