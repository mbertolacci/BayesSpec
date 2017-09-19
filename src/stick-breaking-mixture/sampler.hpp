#ifndef SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_
#define SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/inverse-gamma.hpp"
#include "../random/utils.hpp"
#include "../random/polyagamma.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

class AdaptSpecStickBreakingMixtureSampler : public MixtureSamplerBase<AdaptSpecStickBreakingMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecStickBreakingMixtureSampler> Base;

    AdaptSpecStickBreakingMixtureSampler(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& designMatrix,
        double probMM1,
        double varInflate,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        double tauPriorASquared, double tauPriorNu,
        unsigned int nSplineBases
    ) : Base(x, probMM1, varInflate, componentStart, initialCategories, componentPriors),
        designMatrix_(designMatrix),
        priorMean_(priorMean),
        priorPrecision_(priorPrecision),
        tauPriorASquared_(tauPriorASquared),
        tauPriorNu_(tauPriorNu),
        parameters_(designMatrix_.cols(), nComponents_ - 1),
        tauSquared_(nComponents_ - 1),
        nSplineBases_(nSplineBases) {
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

        if (nSplineBases_ > 0) {
            sampleTau_(rng);
        }
        updateWeights_();
    }

    const Eigen::MatrixXd& getBeta() const {
        return parameters_;
    }

    const Eigen::VectorXd& getTauSquared() const {
        return tauSquared_;
    }

private:
    Eigen::MatrixXd designMatrix_;

    Eigen::MatrixXd priorMean_;
    Eigen::MatrixXd priorPrecision_;
    double tauPriorASquared_;
    double tauPriorNu_;

    Eigen::MatrixXd parameters_;
    Eigen::VectorXd tauSquared_;

    unsigned int nSplineBases_;

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

        Eigen::MatrixXd precision = priorPrecision_.col(component).asDiagonal();
        precision += currentDesignMatrix.transpose() * currentOmega.asDiagonal() * currentDesignMatrix;
        Eigen::MatrixXd precisionCholeskyU = precision.llt().matrixU();

        Eigen::VectorXd z = precisionCholeskyU.transpose().triangularView<Eigen::Lower>().solve(
            currentDesignMatrix.transpose() * currentKappa
            + priorPrecision_.col(component).asDiagonal() * priorMean_.col(component)
        );
        Eigen::VectorXd mean = precisionCholeskyU.triangularView<Eigen::Upper>().solve(z);

        parameters_.col(component) = mean + precisionCholeskyU.triangularView<Eigen::Upper>().solve(
            randNormal(mean.size(), rng)
        );
    }

    template<typename RNG>
    void sampleTau_(RNG& rng) {
        unsigned int splineStartIndex = parameters_.rows() - nSplineBases_;
        for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
            double a = InverseGammaDistribution(
                (tauPriorNu_ + 1.0) / 2.0,
                tauPriorNu_ / tauSquared_[component] + 1 / tauPriorASquared_
            )(rng);
            double residuals = parameters_.col(component).segment(splineStartIndex, nSplineBases_).array().square().sum();

            tauSquared_[component] = InverseGammaDistribution(
                (static_cast<double>(nSplineBases_) + tauPriorNu_) / 2.0,
                residuals / 2.0 + tauPriorNu_ / a
            )(rng);

            for (unsigned int k = 0; k < nSplineBases_; ++k) {
                priorPrecision_(splineStartIndex + k, component) = 1.0 / tauSquared_[component];
            }
        }
    }

    void updateWeights_() {
        Eigen::MatrixXd values = designMatrix_ * parameters_;
        Eigen::MatrixXd beta = (1.0 / (1.0 + (-values).array().exp())).matrix();

        for (unsigned int series = 0; series < x_.cols(); ++series) {
            double prodAccumulator = 1;
            for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
                allWeights_(series, component) = beta(series, component) * prodAccumulator;
                prodAccumulator = prodAccumulator * (1 - beta(series, component));
            }
            allWeights_(series, nComponents_ - 1) = prodAccumulator;
        }
    }
};

}  // namespace bayespec

#endif  // SRC_STICK_BREAKING_MIXTURE_SAMPLER_HPP_
