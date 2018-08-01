#ifndef SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_
#define SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/dirichlet.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

class AdaptSpecIndependentMixtureSampler : public MixtureSamplerBase<AdaptSpecIndependentMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecIndependentMixtureSampler> Base;

    AdaptSpecIndependentMixtureSampler(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        double probMM1,
        double varInflate,
        bool firstCategoryFixed,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::VectorXd& weightsPrior
    ) : Base(x, missingIndices, probMM1, varInflate, firstCategoryFixed, initialCategories, componentPriors),
        weightsPrior_(weightsPrior),
        weights_(nComponents_) {
        weights_.fill(1.0 / static_cast<double>(nComponents_));
        updateWeights_();
    }

    const Eigen::VectorXd& getWeights() const {
        return weights_;
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        weights_ = DirichletDistribution(weightsPrior_ + counts_.cast<double>())(rng);
        updateWeights_();
    }

    double getWeightsLogPrior_() const {
        return ((weightsPrior_.array() - 1) * weights_.array().log()).sum();
    }

private:
    Eigen::VectorXd weightsPrior_;
    Eigen::VectorXd weights_;

    void updateWeights_() {
        allLogWeights_.rowwise() = weights_.array().log().matrix().transpose();
    }
};

}  // namespace bayespec

#endif  // SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_
