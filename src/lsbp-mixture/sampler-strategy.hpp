#ifndef SRC_LSBP_MIXTURE_SAMPLER_STRATEGY_HPP_
#define SRC_LSBP_MIXTURE_SAMPLER_STRATEGY_HPP_

#include <RcppEigen.h>

#include "../random/inverse-gamma.hpp"
#include "../random/truncated-inverse-gamma.hpp"
#include "../random/polyagamma.hpp"
#include "../random/utils.hpp"

namespace bayesspec {

class AdaptSpecLSBPMixtureStrategy {
public:
    void start() {}

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
        unsigned int nComponents = parameters.cols() + 1;
        Eigen::MatrixXd values = designMatrix * parameters;
        Eigen::VectorXi cumulativeCounts(nComponents);
        cumulativeCounts[nComponents - 1] = counts[nComponents - 1];
        for (int component = nComponents - 2; component >= 0; --component) {
            cumulativeCounts[component] = counts[component] + cumulativeCounts[component + 1];
        }

        #pragma omp parallel for
        for (unsigned int component = 0; component < nComponents - 1; ++component) {
            sampleComponentParameters_(
                parameters,
                designMatrix,
                priorMean,
                priorPrecision,
                categories,
                component,
                cumulativeCounts[component],
                values,
                rng
            );
        }

        if (nSplineBases > 0) {
            sampleTau_(
                tauSquared,
                parameters,
                tauPriorNu,
                tauPriorASquared,
                tauPriorUpper,
                nSplineBases,
                rng
            );
        }
    }

    template<typename RNG>
    void sampleComponentParameters_(
        Eigen::MatrixXd& parameters,
        const Eigen::MatrixXd& designMatrix,
        const Eigen::MatrixXd& priorMean,
        const Eigen::MatrixXd& priorPrecision,
        const Eigen::VectorXi& categories,
        unsigned int component,
        unsigned int cumulativeCount,
        const Eigen::MatrixXd& values,
        RNG& rng
    ) {
        Eigen::MatrixXd currentDesignMatrix(cumulativeCount, designMatrix.cols());
        Eigen::VectorXd currentKappa(cumulativeCount);
        Eigen::VectorXd currentOmega(cumulativeCount);

        unsigned int currentIndex = 0;
        for (unsigned int series = 0; series < designMatrix.rows(); ++series) {
            if (categories[series] >= component) {
                currentDesignMatrix.row(currentIndex) = designMatrix.row(series);
                currentKappa[currentIndex] = categories[series] == component ? 0.5 : -0.5;
                currentOmega[currentIndex] = PolyagammaDistribution(values(series, component))(rng);
                ++currentIndex;
            }
        }

        Eigen::MatrixXd precision = priorPrecision.col(component).asDiagonal();
        precision += currentDesignMatrix.transpose() * currentOmega.asDiagonal() * currentDesignMatrix;
        Eigen::MatrixXd precisionCholeskyU = precision.llt().matrixU();

        Eigen::VectorXd z = precisionCholeskyU.transpose().triangularView<Eigen::Lower>().solve(
            currentDesignMatrix.transpose() * currentKappa
            + priorPrecision.col(component).asDiagonal() * priorMean.col(component)
        );
        Eigen::VectorXd mean = precisionCholeskyU.triangularView<Eigen::Upper>().solve(z);

        parameters.col(component) = mean + precisionCholeskyU.triangularView<Eigen::Upper>().solve(
            randNormal(mean.size(), rng)
        );
    }


    template<typename RNG>
    void sampleTau_(
        Eigen::VectorXd& tauSquared,
        const Eigen::MatrixXd& parameters,
        double tauPriorNu,
        double tauPriorASquared,
        double tauPriorUpper,
        unsigned int nSplineBases,
        RNG& rng
    ) {
        unsigned int splineStartIndex = parameters.rows() - nSplineBases;
        for (unsigned int component = 0; component < parameters.cols(); ++component) {
            double a = InverseGammaDistribution(
                (tauPriorNu + 1.0) / 2.0,
                tauPriorNu / tauSquared[component] + 1 / tauPriorASquared
            )(rng);
            double residuals = parameters.col(component).segment(splineStartIndex, nSplineBases).array().square().sum();

            tauSquared[component] = TruncatedInverseGammaDistribution(
                (static_cast<double>(nSplineBases) + tauPriorNu) / 2.0,
                residuals / 2.0 + tauPriorNu / a,
                tauPriorUpper
            )(rng);
        }
    }
};

}  // namespace bayespec

#endif  // SRC_LSBP_MIXTURE_SAMPLER_STRATEGY_HPP_
