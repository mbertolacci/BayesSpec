#ifndef SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_
#define SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/log-beta.hpp"
#include "../random/utils.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

class AdaptSpecDirichletMixtureSampler : public MixtureSamplerBase<AdaptSpecDirichletMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecDirichletMixtureSampler> Base;

    AdaptSpecDirichletMixtureSampler(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        double probMM1,
        double varInflate,
        bool firstCategoryFixed,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        double alphaPriorShape,
        double alphaPriorRate
    ) : Base(x, missingIndices, probMM1, varInflate, firstCategoryFixed, initialCategories, componentPriors),
        alphaPriorShape_(alphaPriorShape),
        alphaPriorRate_(alphaPriorRate),
        logBeta1m_(nComponents_),
        alpha_(1.0) {
        logBeta1m_.fill(std::log(0.5));
        logBeta1m_[nComponents_ - 1] = -std::numeric_limits<double>::infinity();
        updateWeights_();
    }

    const Eigen::VectorXd& getLogBeta1m() const {
        return logBeta1m_;
    }

    const double getAlpha() const {
        return alpha_;
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        unsigned int nRemaining = x_.cols();
        for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
            nRemaining -= counts_[component];
            // Uses the fact that 1 - Beta(a, b) = Beta(b, a)
            logBeta1m_[component] = LogBetaDistribution(nRemaining + alpha_, 1 + counts_[component])(rng);
        }

        alpha_ = randGamma(
            alphaPriorShape_ + nComponents_ - 1,
            1 / (alphaPriorRate_ - logBeta1m_.segment(0, nComponents_ - 1).sum()),
            rng
        );

        updateWeights_();
    }

    double getWeightsLogPrior_() const {
        return (
            // beta prior
            (nComponents_ - 1) * std::log(alpha_)
            + (alpha_ - 1) * logBeta1m_.segment(0, nComponents_ - 1).sum()
            // alpha prior
            + (alphaPriorShape_ - 1) * std::log(alpha_) - alpha_ * alphaPriorRate_
        );
    }

private:
    double alphaPriorShape_;
    double alphaPriorRate_;

    Eigen::VectorXd logBeta1m_;
    double alpha_;

    void updateWeights_() {
        double sumAccumulator = 1;
        for (unsigned int component = 0; component < nComponents_; ++component) {
            allLogWeights_.col(component).fill(
                std::log1p(-std::exp(logBeta1m_[component])) + sumAccumulator
            );
            sumAccumulator = sumAccumulator + logBeta1m_[component];
        }
    }
};

}  // namespace bayespec

#endif  // SRC_DIRICHLET_MIXTURE_SAMPLER_HPP_
