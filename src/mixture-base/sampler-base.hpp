#ifndef SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_
#define SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_

#include <RcppEigen.h>

#include "../utils.hpp"
#include "../random/utils.hpp"
#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"

#include "component-state.hpp"
#include "mixture-base-strategy.hpp"

namespace bayesspec {

template<
    class Instantiation,
    class MixtureBaseStrategyType = MixtureBaseStrategy
>
class MixtureSamplerBase : MixtureBaseStrategyType {
public:
    MixtureSamplerBase(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        const AdaptSpecTuning& componentTuning,
        bool firstCategoryFixed,
        const std::vector<AdaptSpecParameters>& componentStart,
        const Eigen::VectorXi& categoriesStart,
        const std::vector<AdaptSpecPrior>& componentPriors
    ) : x_(x),
        missingIndices_(missingIndices),
        nComponents_(componentPriors.size()),
        firstCategoryFixed_(firstCategoryFixed),
        categories_(categoriesStart),
        allLogWeights_(categoriesStart.size(), nComponents_),
        counts_(nComponents_),
        dataCounts_(nComponents_),
        isWarmedUp_(false) {
        updateCounts_();

        componentStates_.reserve(nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_.emplace_back(
                x_, missingIndices_, componentStart[component], componentPriors[component], componentTuning
            );
        }
    }

    void endWarmUp() {
        isWarmedUp_ = true;
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].endWarmUp();
        }
    }

    bool isWarmedUp() const {
        return isWarmedUp_;
    }

    template<typename RNG>
    void sample(RNG& rng) {
        // Draw new components
        this->sampleComponents_(
            componentStates_,
            categories_.head(x_.cols()),
            dataCounts_,
            rng
        );

        // Draw new categories
        this->sampleCategories_(
            categories_,
            componentStates_,
            allLogWeights_,
            firstCategoryFixed_,
            rng
        );
        updateCounts_();

        // Draw new weights
        static_cast<Instantiation *>(this)->sampleWeights_(rng);
    }

    const Eigen::VectorXi& getCategories() const {
        return categories_;
    }

    const AdaptSpecParameters& getParameters(unsigned int component) const {
        return componentStates_[component].state.parameters;
    }

    double getLogPosterior() const {
        double logPosterior = 0;
        // Likelihood and prior for each component
        for (unsigned int component = 0; component < nComponents_; ++component) {
            logPosterior += componentStates_[component].state.getLogPosterior();
        }

        // Likelihood for categories
        for (unsigned int series = 0; series < categories_.size(); ++series) {
            logPosterior += allLogWeights_(series, categories_[series]);
        }

        // Prior for weights
        logPosterior += static_cast<const Instantiation *>(this)->getWeightsLogPrior_();

        return logPosterior;
    }

    Rcpp::List getParametersAsList() const {
        Rcpp::List componentParameters;
        for (unsigned int i = 0; i < nComponents_; ++i) {
            componentParameters.push_back(getParameters(i).asList());
        }
        Rcpp::List output = static_cast<const Instantiation *>(this)->getWeightsParametersAsList();
        output["components"] = componentParameters;
        output["categories"] = Rcpp::wrap(categories_);
        output["x_missing"] = missingValuesAsList(x_, missingIndices_);
        return output;
    }

    Rcpp::List getComponentStatistics() const {
        Rcpp::List output;
        for (unsigned int i = 0; i < nComponents_; ++i) {
            output.push_back(componentStates_[i].state.getStatistics().asList());
        }
        return output;
    }

    Rcpp::List getComponentWarmUpStatistics() const {
        Rcpp::List output;
        for (unsigned int i = 0; i < nComponents_; ++i) {
            output.push_back(componentStates_[i].state.getWarmUpStatistics().asList());
        }
        return output;
    }

protected:
    Eigen::MatrixXd& x_;
    const std::vector<Eigen::VectorXi>& missingIndices_;
    unsigned int nComponents_;
    bool firstCategoryFixed_;
    std::vector<AdaptSpecMixtureComponentState> componentStates_;

    Eigen::VectorXi categories_;
    Eigen::MatrixXd allLogWeights_;
    Eigen::VectorXi counts_;
    Eigen::VectorXi dataCounts_;

    bool isWarmedUp_;

    void updateCounts_() {
        dataCounts_.fill(0);
        counts_.fill(0);
        for (unsigned int series = 0; series < categories_.size(); ++series) {
            if (series < x_.cols()) {
                ++dataCounts_[categories_[series]];
            }
            ++counts_[categories_[series]];
        }
    }
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_
