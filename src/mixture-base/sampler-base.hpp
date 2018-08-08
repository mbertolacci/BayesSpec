#ifndef SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_
#define SRC_MIXTURE_BASE_SAMPLER_BASE_HPP_

#include <RcppEigen.h>

#include "../utils.hpp"
#include "../random/utils.hpp"
#include "../adaptspec/parameters.hpp"
#include "../adaptspec/prior.hpp"

#include "component-state.hpp"

namespace bayesspec {

template<class Instantiation>
class MixtureSamplerBase {
public:
    MixtureSamplerBase(
        Eigen::MatrixXd& x,
        const std::vector<Eigen::VectorXi>& missingIndices,
        double probMM1,
        double varInflate,
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
        dataCounts_(nComponents_) {
        updateCounts_();

        componentStates_.reserve(nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_.emplace_back(
                x_, missingIndices_, componentStart[component], componentPriors[component], probMM1, varInflate
            );
        }
    }

    void setVarInflate(double newValue) {
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].setVarInflate(newValue);
        }
    }

    template<typename RNG>
    void sample(RNG& rng) {
        // Draw new components
        sampleComponents_(rng);

        // Draw new categories
        sampleCategories_(rng);

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

private:
    template<typename RNG>
    void sampleCategories_(RNG& rng) {
        Eigen::MatrixXd categoryLogLikelihoods(x_.cols(), nComponents_);
        for (unsigned int component = 0; component < nComponents_; ++component) {
            categoryLogLikelihoods.col(component) = componentStates_[component]
                .allLogSegmentLikelihoods
                .leftCols(componentStates_[component].state.parameters.nSegments)
                .rowwise().sum();
        }

        Eigen::MatrixXd categoryLogWeights(allLogWeights_);
        categoryLogWeights.topRows(x_.cols()) += categoryLogLikelihoods;
        categoryLogWeights.colwise() -= categoryLogWeights.rowwise().maxCoeff();

        Eigen::MatrixXd categoryWeights = categoryLogWeights.array().exp().matrix();

        unsigned int series = 0;
        if (firstCategoryFixed_) {
            // NOTE(mgnb): this starts at 1, as the first time-series is fixed
            // to category zero
            series = 1;
        }
        for (; series < categories_.size(); ++series) {
            double u = randUniform(rng) * categoryWeights.row(series).sum();
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

    template<typename RNG>
    void sampleComponents_(RNG& rng) {
        #pragma omp parallel for
        for (unsigned int component = 0; component < nComponents_; ++component) {
            componentStates_[component].sample(
                categories_.head(x_.cols()).array() == static_cast<int>(component),
                dataCounts_[component],
                rng
            );
        }
    }

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
