#ifndef SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_
#define SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/dirichlet.hpp"
#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"

#include "../mixture-base/component-state.hpp"

namespace bayesspec {

class AdaptSpecIndependentMixtureSampler {
public:
    AdaptSpecIndependentMixtureSampler(
        const Eigen::MatrixXd& x,
        double probMM1,
        const std::vector<AdaptSpecParameters>& componentStart,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::VectorXd weightsPrior
    ) : x_(x),
        nComponents_(componentPriors.size()),
        weightsPrior_(weightsPrior),
        categories_(x.cols()),
        weights_(nComponents_),
        counts_(nComponents_) {

        for (unsigned int series = 0; series < x.cols(); ++series) {
            categories_[series] = rng.randint(0, nComponents_ - 1);
        }
        updateCounts_();

        componentStates_.reserve(nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_.emplace_back(
                x_, componentStart[component], componentPriors[component], probMM1
            );
        }

        weights_.fill(1 / static_cast<double>(nComponents_));
    }

    void sample() {
        // Draw new categories
        sampleCategories_();

        // Draw new weights
        sampleWeights_();

        // Draw new components
        sampleComponents_();
    }

    const Eigen::VectorXi& getCategories() const {
        return categories_;
    }

    const Eigen::VectorXd& getWeights() const {
        return weights_;
    }

    const AdaptSpecParameters& getParameters(unsigned int component) const {
        return componentStates_[component].state.parameters;
    }

private:
    const Eigen::MatrixXd& x_;
    unsigned int nComponents_;
    std::vector<AdaptSpecMixtureComponentState> componentStates_;
    Eigen::VectorXd weightsPrior_;

    Eigen::VectorXi categories_;
    Eigen::VectorXd weights_;
    Eigen::VectorXi counts_;

    void sampleWeights_() {
        weights_ = DirichletDistribution(weightsPrior_ + counts_.cast<double>())(rng.engine_);
    }

    void sampleCategories_() {
        Eigen::MatrixXd categoryLogLikelihoods(x_.cols(), nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            categoryLogLikelihoods.col(component) = componentStates_[component]
                .allLogSegmentLikelihoods
                .leftCols(componentStates_[component].state.parameters.nSegments)
                .rowwise().sum();
        }
        categoryLogLikelihoods.colwise() -= categoryLogLikelihoods.rowwise().maxCoeff();

        Eigen::MatrixXd categoryWeights = (
            categoryLogLikelihoods.array().exp().rowwise() * weights_.transpose().array()
        ).matrix();

        for (unsigned int series = 0; series < categories_.size(); ++series) {
            double u = rng.randu() * categoryWeights.row(series).sum();
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

    void sampleComponents_() {
        #pragma omp parallel for
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].sample(
                categories_.array() == static_cast<int>(component),
                counts_[component]
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

#endif  // SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_
