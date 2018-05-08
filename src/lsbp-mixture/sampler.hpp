#ifndef SRC_LSBP_SAMPLER_HPP_
#define SRC_LSBP_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../random/inverse-gamma.hpp"
#include "../random/utils.hpp"
#include "../random/polyagamma.hpp"
#include "../mixture-base/sampler-base.hpp"

namespace bayesspec {

// log(1 + exp(x))
double log1pexp(double x) {
    // At this point, the difference between log(1 + exp(x)) and x is below
    // machine precision, and this prevents overflow
    if (x > 36.04365) return x;
    return std::log1p(std::exp(x));
}

class AdaptSpecLogisticStickBreakingPriorMixtureSampler : public MixtureSamplerBase<AdaptSpecLogisticStickBreakingPriorMixtureSampler> {
public:
    typedef MixtureSamplerBase<AdaptSpecLogisticStickBreakingPriorMixtureSampler> Base;

    AdaptSpecLogisticStickBreakingPriorMixtureSampler(
        const Eigen::MatrixXd& x,
        const Eigen::MatrixXd& designMatrix,
        double probMM1,
        double varInflate,
        bool firstCategoryFixed,
        const Eigen::VectorXi& initialCategories,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        double tauPriorASquared, double tauPriorNu,
        unsigned int nSplineBases
    ) : Base(x, probMM1, varInflate, firstCategoryFixed, initialCategories, componentPriors),
        designMatrix_(designMatrix),
        priorMean_(priorMean),
        priorPrecision_(priorPrecision),
        tauPriorASquared_(tauPriorASquared),
        tauPriorNu_(tauPriorNu),
        parameters_(designMatrix_.cols(), nComponents_ - 1),
        tauSquared_(nComponents_ - 1),
        nSplineBases_(nSplineBases) {
        parameters_.fill(0);
        tauSquared_.fill(1);
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

    double getWeightsLogPrior_() const {
        return (
            priorPrecision_.array().log().sum() / 2.0
            - ((parameters_ - priorMean_).array().square() * priorPrecision_.array()).sum() / 2.0
            - (tauPriorNu_ + 1) / 2.0 * (1.0 + tauSquared_.array() / (tauPriorNu_ * tauPriorASquared_)).log().sum()
        );
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
        Eigen::MatrixXd values = (designMatrix_ * parameters_).array().matrix();

        for (unsigned int series = 0; series < x_.cols(); ++series) {
            double sumAccumulator = 0;
            for (unsigned int component = 0; component < nComponents_ - 1; ++component) {
                allLogWeights_(series, component) = (
                    -log1pexp(-values(series, component)) + sumAccumulator
                );
                sumAccumulator = sumAccumulator - log1pexp(values(series, component));
            }
            allLogWeights_(series, nComponents_ - 1) = sumAccumulator;
        }
    }
};

}  // namespace bayespec

#endif  // SRC_LSBP_SAMPLER_HPP_
