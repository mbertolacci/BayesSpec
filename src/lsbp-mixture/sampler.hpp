#ifndef SRC_LSBP_SAMPLER_HPP_
#define SRC_LSBP_SAMPLER_HPP_

#include <RcppEigen.h>

#include "../mixture-base/sampler-base.hpp"
#include "sampler-strategy.hpp"

namespace bayesspec {

// log(1 + exp(x))
double log1pexp(double x) {
    // At this point, the difference between log(1 + exp(x)) and x is below
    // machine precision, and this prevents overflow
    if (x > 36.04365) return x;
    return std::log1p(std::exp(x));
}

template<
    typename AdaptSpecLSBPMixtureStrategyType,
    typename MixtureBaseStrategyType
>
class AdaptSpecLSBPMixtureSamplerBase
    : public AdaptSpecLSBPMixtureStrategyType,
      public MixtureSamplerBase<
        AdaptSpecLSBPMixtureSamplerBase<
            AdaptSpecLSBPMixtureStrategyType,
            MixtureBaseStrategyType
        >,
        MixtureBaseStrategyType
    >
{
public:
    typedef MixtureSamplerBase<
        AdaptSpecLSBPMixtureSamplerBase<
            AdaptSpecLSBPMixtureStrategyType,
            MixtureBaseStrategyType
        >,
        MixtureBaseStrategyType
    > Base;

    AdaptSpecLSBPMixtureSamplerBase(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        const Eigen::MatrixXd& designMatrix,
        const AdaptSpecTuning& componentTuning,
        bool firstCategoryFixed,
        const Eigen::MatrixXd& betaStart,
        const Eigen::VectorXd& tauSquaredStart,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& categoriesStart,
        const std::vector<AdaptSpecPrior>& componentPriors,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        double tauPriorASquared, double tauPriorNu, double tauPriorUpper,
        unsigned int nSplineBases
    ) : Base(
            x, missingIndices,
            componentTuning, firstCategoryFixed,
            componentStart, categoriesStart,
            componentPriors
        ),
        designMatrix_(designMatrix),
        priorMean_(priorMean),
        priorPrecision_(priorPrecision),
        tauPriorASquared_(tauPriorASquared),
        tauPriorNu_(tauPriorNu),
        tauPriorUpper_(tauPriorUpper),
        parameters_(betaStart),
        tauSquared_(tauSquaredStart),
        nSplineBases_(nSplineBases) {
        updateWeights_();
    }

    template<typename RNG>
    void sampleWeights_(RNG& rng) {
        this->sampleLSBPWeights_(
            parameters_,
            tauSquared_,
            designMatrix_,
            priorMean_,
            priorPrecision_,
            this->categories_,
            this->counts_,
            tauPriorNu_,
            tauPriorASquared_,
            tauPriorUpper_,
            nSplineBases_,
            rng
        );

        if (nSplineBases_ > 0) {
            updatePriorPrecision_();
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

    Rcpp::List getWeightsParametersAsList() const {
        Rcpp::List output;
        output["beta"] = Rcpp::wrap(parameters_);
        output["tau_squared"] = Rcpp::wrap(tauSquared_);
        return output;
    }

private:
    Eigen::MatrixXd designMatrix_;

    Eigen::MatrixXd priorMean_;
    Eigen::MatrixXd priorPrecision_;
    double tauPriorASquared_;
    double tauPriorNu_;
    double tauPriorUpper_;

    Eigen::MatrixXd parameters_;
    Eigen::VectorXd tauSquared_;

    unsigned int nSplineBases_;

    void updatePriorPrecision_() {
        unsigned int splineStartIndex = parameters_.rows() - nSplineBases_;
        for (unsigned int component = 0; component < this->nComponents_ - 1; ++component) {
            for (unsigned int k = 0; k < nSplineBases_; ++k) {
                priorPrecision_(splineStartIndex + k, component) = 1.0 / tauSquared_[component];
            }
        }
    }

    void updateWeights_() {
        Eigen::MatrixXd values = (designMatrix_ * parameters_).array().matrix();

        for (unsigned int series = 0; series < designMatrix_.rows(); ++series) {
            double sumAccumulator = 0;
            for (unsigned int component = 0; component < this->nComponents_ - 1; ++component) {
                this->allLogWeights_(series, component) = (
                    -log1pexp(-values(series, component)) + sumAccumulator
                );
                sumAccumulator = sumAccumulator - log1pexp(values(series, component));
            }
            this->allLogWeights_(series, this->nComponents_ - 1) = sumAccumulator;
        }
    }
};

typedef AdaptSpecLSBPMixtureSamplerBase<
    AdaptSpecLSBPMixtureStrategy,
    MixtureBaseStrategy
> AdaptSpecLSBPMixtureSampler;

}  // namespace bayespec

#endif  // SRC_LSBP_SAMPLER_HPP_
