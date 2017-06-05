#ifndef SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_
#define SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/polyagamma.hpp"
#include "../random/utils.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

class AdaptSpecStickBreakingMixtureSampler : public MixtureSamplerBase<AdaptSpecStickBreakingMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecStickBreakingMixtureSampler> Base;

    AdaptSpecStickBreakingMixtureSampler(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& designMatrix,
        double probMM1,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::VectorXd& priorMean,
        const Eigen::MatrixXd& priorPrecision
    ) : Base(x, probMM1, componentStart, initialCategories, componentPriors),
        designMatrix_(designMatrix),
        priorMean_(priorMean),
        priorPrecision_(priorPrecision),
        parameters_(designMatrix_.cols(), nComponents_ - 1) {
        parameters_.fill(0);
        updateWeights_();
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        Eigen::MatrixXd values = designMatrix_ * parameters_;
        Eigen::VectorXi cumulativeCounts(nComponents_);
        cumulativeCounts[nComponents_ - 1] = counts_[nComponents_ - 1];
        for (int component = nComponents_ - 2; component >= 0; --component) {
            cumulativeCounts[component] = counts_[component] + cumulativeCounts[component + 1];
        }

        #pragma omp parallel for
        for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
            sampleComponentParameters_(component, cumulativeCounts[component], values, rng);
        }

        updateWeights_();
    }

private:
    Eigen::MatrixXd designMatrix_;

    Eigen::VectorXd priorMean_;
    Eigen::MatrixXd priorPrecision_;

    Eigen::MatrixXd parameters_;

    template<typename RNG>
    void sampleComponentParameters_(unsigned int component, unsigned int cumulativeCount, const Eigen::MatrixXd& values, RNG& rng) {
        Eigen::MatrixXd currentDesignMatrix(cumulativeCount, designMatrix_.cols());
        Eigen::VectorXd currentKappa(cumulativeCount);
        Eigen::VectorXd currentOmega(cumulativeCount);

        unsigned int currentIndex = 0;
        for (unsigned int series = 0; series < x_.cols(); ++series) {
            if (categories_[series] >= component) {
                currentDesignMatrix.row(currentIndex) = designMatrix_.row(series);
                currentKappa[currentIndex] = categories_[series] == component ? 0.5 : -0.5;
                currentOmega[currentIndex] = PolyagammaDistribution(values(series, component))(rng);
                ++currentIndex;
            }
        }

        Eigen::MatrixXd precisionCholeskyU = (
            currentDesignMatrix.transpose() * currentOmega.asDiagonal() * currentDesignMatrix
            + priorPrecision_
        ).llt().matrixU();

        Eigen::VectorXd z = precisionCholeskyU.transpose().triangularView<Eigen::Lower>().solve(
            currentDesignMatrix.transpose() * currentKappa
            + priorPrecision_ * priorMean_
        );
        Eigen::VectorXd mean = precisionCholeskyU.triangularView<Eigen::Upper>().solve(z);

        parameters_.col(component) = mean + precisionCholeskyU.triangularView<Eigen::Upper>().solve(
            randNormal(mean.size(), rng)
        );
    }

    void updateWeights_() {
        Eigen::MatrixXd values = designMatrix_ * parameters_;
        Eigen::MatrixXd beta = (1.0 / (1.0 + (-values).array().exp())).matrix();

        for (unsigned int series = 0; series < x_.cols(); ++series) {
            double prodAccumulator = 1;
            for (unsigned int component = 0; component < nComponents_; ++component) {
                allWeights_(series, component) = beta(series, component) * prodAccumulator;
                prodAccumulator = prodAccumulator * (1 - beta(series, component));
            }
        }
    }
};

}  // namespace bayespec

#endif  // SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_
