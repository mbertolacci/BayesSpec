#ifndef SRC_LSBP_INDEPENDENT_PROPOSAL_HPP_
#define SRC_LSBP_INDEPENDENT_PROPOSAL_HPP_

#include <RcppEigen.h>
#include "../random/utils.hpp"

namespace bayesspec {

class LSBPIndependentProposal  {
public:
    LSBPIndependentProposal(
        unsigned int component,
        const Eigen::VectorXi& categories,
        const Eigen::VectorXd& priorMean,
        const Eigen::VectorXd& priorPrecision,
        const Eigen::MatrixXd& designMatrix,
        const Eigen::VectorXd& start
    ) {
        Eigen::VectorXd pastEstimate(start);
        Eigen::VectorXd currentEstimate(start);
        double tolerance = std::sqrt(Eigen::NumTraits<double>::epsilon());

        unsigned int count = 0;
        for (unsigned int i = 0; i < categories.size(); ++i) {
            if (categories[i] >= component) ++count;
        }
        if (count == 0) {
            mean_ = priorMean;
            precisionLLT_.compute(priorPrecision.asDiagonal());
            return;
        }

        Eigen::VectorXd kappa(count);
        Eigen::MatrixXd componentDesignMatrix(count, designMatrix.cols());
        unsigned int currentIndex = 0;
        for (unsigned int i = 0; i < categories.size(); ++i) {
            if (categories[i] >= component) {
                kappa[currentIndex] = categories[i] == component ? 0.5 : -0.5;
                componentDesignMatrix.row(currentIndex) = designMatrix.row(i);
                ++currentIndex;
            }
        }

        Eigen::VectorXd d = (
            componentDesignMatrix.transpose() * kappa
            + priorPrecision.asDiagonal() * priorMean
        );

        unsigned int iteration = 0;
        for (; iteration < MAX_ITERATIONS; ++iteration) {
            Eigen::VectorXd currentValues = componentDesignMatrix * currentEstimate;
            Eigen::VectorXd currentOmega = 0.5 * (
                (0.5 * currentValues).array().tanh() / currentValues.array()
            ).matrix();

            Eigen::MatrixXd precisionCurrent = priorPrecision.asDiagonal();
            precisionCurrent += componentDesignMatrix.transpose() * currentOmega.asDiagonal() * componentDesignMatrix;
            precisionLLT_.compute(precisionCurrent);
            currentEstimate = precisionLLT_.solve(d);

            if ((pastEstimate - currentEstimate).squaredNorm() < tolerance) {
                break;
            }
            pastEstimate = currentEstimate;
        }

        mean_ = currentEstimate;

        Eigen::VectorXd p = (1.0 / (1.0 + (-componentDesignMatrix * currentEstimate).array().exp())).matrix();
        Eigen::MatrixXd hessianPrecision = priorPrecision.asDiagonal();
        hessianPrecision += (
            componentDesignMatrix.transpose() *
            (p.array() * (1.0 - p.array())).matrix().asDiagonal() *
            componentDesignMatrix
        );

        precisionLLT_.compute(hessianPrecision);
    }

    double logDensity(const Eigen::VectorXd& x) {
        double output = 0;
        // HACK(mgnb): Eigen has no quick way to get log determinant of
        // triangular matrix
        for (unsigned int i = 0; i < x.size(); ++i) {
            output += std::log(precisionLLT_.matrixL()(i, i));
        }
        output -= 0.5 * (precisionLLT_.matrixU() * (x - mean_)).array().square().sum();

        return output;
    }

    template<typename RNG>
    Eigen::VectorXd sample(RNG& rng) {
        return mean_ + precisionLLT_.matrixU().solve(randNormal(mean_.size(), rng));
    }

private:
    Eigen::VectorXd mean_;
    Eigen::LLT<Eigen::MatrixXd> precisionLLT_;
    const unsigned int MAX_ITERATIONS = 1000;
};


}  // namespace bayespec

#endif  // SRC_LSBP_INDEPENDENT_PROPOSAL_HPP_
