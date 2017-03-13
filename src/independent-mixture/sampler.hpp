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
        const Eigen::MatrixXd& x,
        double probMM1,
        const std::vector<AdaptSpecParameters>& componentStart,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::VectorXd weightsPrior
    ) : Base(x, probMM1, componentStart, componentPriors),
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

private:
    Eigen::VectorXd weightsPrior_;
    Eigen::VectorXd weights_;

    void updateWeights_() {
        allWeights_.rowwise() = weights_.transpose();
    }
};

}  // namespace bayespec

#endif  // SRC_INDEPENDENT_MIXTURE_SAMPLER_HPP_
