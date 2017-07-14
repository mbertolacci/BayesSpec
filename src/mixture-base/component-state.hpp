#ifndef SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_
#define SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_

#include <RcppEigen.h>

#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"
#include "../adaptspec/state.hpp"

namespace bayesspec {

class AdaptSpecMixtureComponentState {
public:
    std::vector<Eigen::MatrixXd> allPeriodograms;
    Eigen::MatrixXd allLogSegmentLikelihoods;

    AdaptSpecState state;

    AdaptSpecMixtureComponentState(
        const Eigen::MatrixXd& x,
        const AdaptSpecParameters& start,
        const AdaptSpecPrior& prior,
        double probMM1,
        double varInflate
    ) : state(start, x, prior, probMM1, varInflate),
        x_(x) {
        allPeriodograms.resize(prior.nSegmentsMax);
        allLogSegmentLikelihoods.resize(x.cols(), prior.nSegmentsMax);

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

    template<typename RNG>
    void sample(const Eigen::Array<bool, Eigen::Dynamic, 1>& isComponent, unsigned int count, RNG& rng) {
        Eigen::MatrixXd thisX(x_.rows(), count);

        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            state.periodogram[segment].resize(state.periodogram[segment].rows(), count);
        }

        unsigned int currentIndex = 0;
        for (unsigned int series = 0; series < isComponent.size(); ++series) {
            if (isComponent[series]) {
                thisX.col(currentIndex) = x_.col(series);
                for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
                    state.periodogram[segment].col(currentIndex) = allPeriodograms[segment].col(series);
                }
                ++currentIndex;
            }
        }
        state.x = &thisX;
        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            state.updateSegmentFit(segment);
        }

        AdaptSpecParameters oldParameters = state.parameters;
        Eigen::VectorXd oldSegmentLengths = state.segmentLengths;
        state.sample(rng);

        std::vector<Eigen::MatrixXd> oldAllPeriodograms = allPeriodograms;
        Eigen::MatrixXd oldAllLogSegmentLikelihoods = allLogSegmentLikelihoods;

        for (unsigned int segment = 0; segment < state.parameters.nSegments; ++segment) {
            bool sameSegment = false;
            bool sameBeta = false;

            for (unsigned int segmentOld = 0; segmentOld < oldParameters.nSegments; ++segmentOld) {
                if (state.parameters.cutPoints[segment] == oldParameters.cutPoints[segmentOld]
                    && state.segmentLengths[segment] == oldSegmentLengths[segmentOld]) {
                    sameSegment = true;
                    allPeriodograms[segment] = oldAllPeriodograms[segmentOld];
                    if (state.parameters.beta.row(segment) == oldParameters.beta.row(segmentOld)) {
                        allLogSegmentLikelihoods.col(segment) = oldAllLogSegmentLikelihoods.col(segmentOld);
                        sameBeta = true;
                    }
                } else {
                    allPeriodograms[segment].resize(state.periodogram[segment].rows(), x_.cols());
                }
            }

            if (!sameSegment) {
                unsigned int currentIndex = 0;
                for (unsigned int series = 0; series < x_.cols(); ++series) {
                    if (isComponent[series]) {
                        allPeriodograms[segment].col(series) = state.periodogram[segment].col(currentIndex);
                        ++currentIndex;
                    } else {
                        allPeriodograms[segment].col(series) = AdaptSpecUtils::calculatePeriodogram(
                            x_.col(series),
                            state.parameters.cutPoints[segment],
                            state.segmentLengths[segment]
                        );
                    }
                }
            }

            if (!sameBeta) {
                allLogSegmentLikelihoods.col(segment) = logWhittleLikelihood(
                    state.nu[segment] * state.parameters.beta.row(segment).transpose(),
                    allPeriodograms[segment],
                    state.segmentLengths[segment]
                );
            }
        }


    }

private:
    const Eigen::MatrixXd& x_;
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_COMPONENT_STATE_HPP_
