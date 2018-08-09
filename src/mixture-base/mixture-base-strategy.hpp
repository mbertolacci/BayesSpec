#ifndef SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_HPP_
#define SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_HPP_

#include <RcppEigen.h>
#include "component-state.hpp"

namespace bayesspec {

class MixtureBaseStrategy {
protected:
    template<typename RNG>
    void sampleComponents_(
        std::vector<AdaptSpecMixtureComponentState>& componentStates,
        const Eigen::VectorXi& categories,
        const Eigen::VectorXi& counts,
        RNG& rng
    ) {
        #pragma omp parallel for
        for (unsigned int component = 0; component < componentStates.size(); ++component) {
            componentStates[component].sample(
                categories.array() == static_cast<int>(component),
                counts[component],
                rng
            );
        }
    }

    template<typename RNG>
    void sampleCategories_(
        Eigen::VectorXi& categories,
        const std::vector<AdaptSpecMixtureComponentState>& componentStates,
        const Eigen::MatrixXd& allLogWeights,
        bool firstCategoryFixed,
        RNG& rng
    ) {
        unsigned int nComponents = componentStates.size();
        unsigned int nTimeSeries = componentStates[0].allLogSegmentLikelihoods.rows();

        Eigen::MatrixXd categoryLogLikelihoods(nTimeSeries, nComponents);
        for (unsigned int component = 0; component < nComponents; ++component) {
            categoryLogLikelihoods.col(component) = componentStates[component]
                .allLogSegmentLikelihoods
                .leftCols(componentStates[component].state.parameters.nSegments)
                .rowwise().sum();
        }

        Eigen::MatrixXd categoryLogWeights(allLogWeights);
        categoryLogWeights.topRows(nTimeSeries) += categoryLogLikelihoods;
        categoryLogWeights.colwise() -= categoryLogWeights.rowwise().maxCoeff();

        Eigen::MatrixXd categoryWeights = categoryLogWeights.array().exp().matrix();

        unsigned int series = 0;
        if (firstCategoryFixed) {
            // NOTE(mgnb): this starts at 1, as the first time-series is fixed
            // to category zero
            series = 1;
        }
        for (; series < categories.size(); ++series) {
            double u = randUniform(rng) * categoryWeights.row(series).sum();
            for (unsigned int component = 0; component < nComponents; ++component) {
                u -= categoryWeights(series, component);
                if (u <= 0) {
                    categories[series] = component;
                    break;
                }
            }
        }
    }
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_HPP_
