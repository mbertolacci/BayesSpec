#ifndef SRC_LSBP_SAMPLER_MPI_HPP_
#define SRC_LSBP_SAMPLER_MPI_HPP_

#include <RcppEigen.h>

#include "sampler.hpp"
#include "../mixture-base/mixture-base-strategy-mpi.hpp"
#include "../mpi.hpp"

namespace bayesspec {

class AdaptSpecLSBPMixtureStrategyMPI : public AdaptSpecLSBPMixtureStrategy {
protected:
    template<typename RNG>
    void sampleLSBPWeights_(
        Eigen::MatrixXd& parameters,
        Eigen::VectorXd& tauSquared,
        const Eigen::MatrixXd& designMatrix,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        const Eigen::VectorXi& categories,
        const Eigen::VectorXi& counts,
        double tauPriorNu,
        double tauPriorASquared,
        double tauPriorUpper,
        unsigned int nSplineBases,
        RNG& rng
    ) {
        if (MPI::rank() == 0) {
            AdaptSpecLSBPMixtureStrategy::sampleLSBPWeights_(
                parameters,
                tauSquared,
                designMatrix,
                priorMean,
                priorPrecision,
                categories,
                counts,
                tauPriorNu,
                tauPriorASquared,
                tauPriorUpper,
                nSplineBases,
                rng
            );
        }

        MPI::broadcast(parameters, 0);
        MPI::broadcast(tauSquared, 0);
    }
};

typedef AdaptSpecLSBPMixtureSamplerBase<
    AdaptSpecLSBPMixtureStrategyMPI,
    MixtureBaseStrategyMPI
> AdaptSpecLSBPMixtureSamplerMPI;

}  // namespace bayespec

#endif  // SRC_LSBP_SAMPLER_MPI_HPP_
