#ifndef SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_MPI_HPP_
#define SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_MPI_HPP_

#include <RcppEigen.h>
#include "component-state.hpp"
#include "mixture-base-strategy.hpp"
#include "../mpi.hpp"

namespace bayesspec {

class MixtureBaseStrategyMPI : MixtureBaseStrategy {
protected:
    template<typename RNG>
    void sampleComponents_(
        std::vector<AdaptSpecMixtureComponentState>& componentStates,
        const Eigen::VectorXi& categories,
        const Eigen::VectorXi& counts,
        RNG& rng
    ) {
        unsigned int nComponents = componentStates.size();
        int rank = MPI::rank();
        int size = MPI::size();
        std::vector<unsigned int> myComponents;
        for (unsigned int component = 0; component < nComponents; ++component) {
            if (component % size == rank) {
                myComponents.push_back(component);
            }
        }

        #pragma omp parallel for
        for (unsigned int i = 0; i < myComponents.size(); ++i) {
            unsigned int component = myComponents[i];
            componentStates[component].sample(
                categories.array() == static_cast<int>(component),
                counts[component],
                rng
            );
        }

        // HACK(mgnb): should use allgather for this
        for (unsigned int component = 0; component < nComponents; ++component) {
            broadcastComponentState_(
                componentStates[component],
                categories.array() == static_cast<int>(component),
                component % size
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
        if (MPI::rank() == 0) {
            MixtureBaseStrategy::sampleCategories_(
                categories,
                componentStates,
                allLogWeights,
                firstCategoryFixed,
                rng
            );
        }

        MPI::broadcast(categories, 0);
    }

private:
    void broadcastComponentState_(
        AdaptSpecMixtureComponentState& componentState,
        const AdaptSpecMixtureComponentState::BoolArray& isComponent,
        int senderRank
    ) {
        // NOTE(mgnb): this doesn't sync the full state, just the parts relevant
        // to other steps in the sampler, so it may need to change if those
        // steps change

        int rank = MPI::rank();

        MPI::broadcast(componentState.state.parameters.nSegments, senderRank);
        MPI::broadcast(componentState.state.parameters.beta, senderRank);
        MPI::broadcast(componentState.state.parameters.tauSquared, senderRank);
        MPI::broadcast(componentState.state.parameters.cutPoints, senderRank);
        MPI::broadcast(componentState.state.logSegmentLikelihood, senderRank);
        MPI::broadcast(componentState.state.logSegmentPrior, senderRank);
        MPI::broadcast(componentState.state.logPriorCutPoints, senderRank);

        for (unsigned int segment = 0; segment < componentState.state.parameters.nSegments; ++segment) {
            MPI::broadcast(componentState.allPeriodograms[segment], senderRank, true);
        }
        MPI::broadcast(componentState.allLogSegmentLikelihoods, senderRank);

        for (unsigned int timeSeries = 0; timeSeries < isComponent.size(); ++timeSeries) {
            if (isComponent[timeSeries]) {
                unsigned int nMissing = componentState.missingIndices[timeSeries].size();
                if (nMissing == 0) continue;

                Eigen::VectorXd xMissing(nMissing);
                if (rank == senderRank) {
                    for (unsigned int i = 0; i < nMissing; ++i) {
                        xMissing[i] = componentState.x->operator()(
                            componentState.missingIndices[timeSeries][i],
                            timeSeries
                        );
                    }
                }
                MPI::broadcast(xMissing, senderRank);
                if (rank != senderRank) {
                    for (unsigned int i = 0; i < nMissing; ++i) {
                        componentState.x->operator()(
                            componentState.missingIndices[timeSeries][i],
                            timeSeries
                        ) = xMissing[i];
                    }
                }
            }
        }
    }
};

}  // namespace bayespec

#endif  // SRC_MIXTURE_BASE_MIXTURE_BASE_STRATEGY_MPI_HPP_
