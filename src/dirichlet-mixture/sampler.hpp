#ifndef SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_
#define SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/beta.hpp"
#include "../random/utils.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

class AdaptSpecDirichletMixtureSampler : public MixtureSamplerBase<AdaptSpecDirichletMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecDirichletMixtureSampler> Base;

    AdaptSpecDirichletMixtureSampler(
        const Eigen::MatrixXd& x,
        double probMM1,
        double varInflate,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        double alphaPriorShape,
        double alphaPriorRate
    ) : Base(x, probMM1, varInflate, componentStart, initialCategories, componentPriors),
        alphaPriorShape_(alphaPriorShape),
        alphaPriorRate_(alphaPriorRate),
        beta_(nComponents_),
        alpha_(1.0) {
        beta_.fill(0.5);
        beta_[nComponents_ - 1] = 1;
        updateWeights_();
    }

    const Eigen::VectorXd& getBeta() const {
        return beta_;
    }

    const double getAlpha() const {
        return alpha_;
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        unsigned int nRemaining = x_.cols();
        for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
            nRemaining -= counts_[component];
            beta_[component] = BetaDistribution(1 + counts_[component], nRemaining + alpha_)(rng);
        }

        alpha_ = randGamma(
            alphaPriorShape_ + nComponents_ - 1,
            alphaPriorRate_ - (1.0 - beta_.segment(0, nComponents_ - 1).array()).log().sum(),
            rng
        );

        updateWeights_();
    }

private:
    double alphaPriorShape_;
    double alphaPriorRate_;

    Eigen::VectorXd beta_;
    double alpha_;

    void updateWeights_() {
        double prodAccumulator = 1;
        for (unsigned int component = 0; component < nComponents_; ++component) {
            allWeights_.col(component).fill(beta_[component] * prodAccumulator);
            prodAccumulator = prodAccumulator * (1 - beta_[component]);
        }
    }
};

}  // namespace bayespec

#endif  // SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_
