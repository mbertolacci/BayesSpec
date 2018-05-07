#ifndef SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_
#define SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_

#include <RcppEigen.h>

#include "../random/utils.hpp"
#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"

#include "component-state.hpp"

namespace bayesspec {

template<class Instantiation>
class MixtureSamplerBase {
public:
    MixtureSamplerBase(
        const Eigen::MatrixXd& x,
        double probMM1,
        double varInflate,
        bool firstCategoryFixed,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors
    ) : x_(x),
        nComponents_(componentPriors.size()),
        firstCategoryFixed_(firstCategoryFixed),
        categories_(initialCategories),
        allWeights_(x.cols(), nComponents_),
        counts_(nComponents_) {
        updateCounts_();

        componentStates_.reserve(nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_.emplace_back(
                x_, componentStart[component], componentPriors[component], probMM1, varInflate
            );
        }
    }

    void setVarInflate(double newValue) {
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].setVarInflate(newValue);
        }
    }

    template<typename RNG>
    void sample(RNG& rng) {
        // Draw new components
        sampleComponents_(rng);

        // Draw new categories
        sampleCategories_(rng);

        // Draw new weights
        static_cast<Instantiation *>(this)->sampleWeights_(rng);
    }

    const Eigen::VectorXi& getCategories() const {
        return categories_;
    }

    const AdaptSpecParameters& getParameters(unsigned int component) const {
        return componentStates_[component].state.parameters;
    }

protected:
    const Eigen::MatrixXd& x_;
    unsigned int nComponents_;
    bool firstCategoryFixed_;
    std::vector<AdaptSpecMixtureComponentState> componentStates_;

    Eigen::VectorXi categories_;
    Eigen::MatrixXd allWeights_;
    Eigen::VectorXi counts_;

private:
    template<typename RNG>
    void sampleCategories_(RNG& rng) {
        Eigen::MatrixXd categoryLogLikelihoods(x_.cols(), nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            categoryLogLikelihoods.col(component) = componentStates_[component]
                .allLogSegmentLikelihoods
                .leftCols(componentStates_[component].state.parameters.nSegments)
                .rowwise().sum();
        }
        categoryLogLikelihoods.colwise() -= categoryLogLikelihoods.rowwise().maxCoeff();

        Eigen::MatrixXd categoryWeights = (
            categoryLogLikelihoods.array().exp() * allWeights_.array()
        ).matrix();

        unsigned int series = 0;
        if (firstCategoryFixed_) {
            // NOTE(mgnb): this starts at 1, as the first time-series is fixed
            // to category zero
            series = 1;
        }
        for (; series < categories_.size(); ++series) {
            double u = randUniform(rng) * categoryWeights.row(series).sum();
            for (unsigned int component = 0; component < nComponents_; ++component) {
                u -= categoryWeights(series, component);
                if (u <= 0) {
                    categories_[series] = component;
                    break;
                }
            }
        }

        updateCounts_();
    }

    template<typename RNG>
    void sampleComponents_(RNG& rng) {
        #pragma omp parallel for
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].sample(
                categories_.array() == static_cast<int>(component),
                counts_[component],
                rng
            );
        }
    }

    void updateCounts_() {
        counts_.fill(0);
        for (unsigned int series = 0; series < categories_.size(); ++series) {
            ++counts_[categories_[series]];
        }
    }
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_
