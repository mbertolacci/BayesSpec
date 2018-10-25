#ifndef SRC_ADAPTSPEC_HMC_HPP_
#define SRC_ADAPTSPEC_HMC_HPP_

#include <RcppEigen.h>

#include "../random/utils.hpp"

namespace bayesspec {

template<typename RNG>
Eigen::VectorXd sampleBetaHmc(
    const Eigen::VectorXd& betaCurrent,
    unsigned int n,
    const Eigen::MatrixXd& periodogram,
    const Eigen::MatrixXd& nu,
    double sigmaSquaredAlpha,
    double tauSquared,
    int lMin,
    int lMax,
    double epsilonMin,
    double epsilonMax,
    RNG& rng
) {
    using Eigen::VectorXd;

    BetaFunctor betaFunctor(
        n,
        periodogram,
        nu,
        sigmaSquaredAlpha,
        tauSquared
    );

    // Randomise HMC parameters
    int l = randInteger(lMin, lMax, rng);
    double epsilon = epsilonMin + (epsilonMax - epsilonMin) * randUniform(rng);

    betaFunctor.setBeta(betaCurrent);
    double currentU = betaFunctor.value();
    VectorXd gradientCurrent(betaCurrent.size());
    betaFunctor.gradient(gradientCurrent);
    VectorXd currentZ = randNormal(betaCurrent.size(), rng);
    VectorXd currentP = betaFunctor.fisherLLT().matrixL() * currentZ;

    // Starting values
    VectorXd beta(betaCurrent);
    VectorXd gradient(gradientCurrent);
    VectorXd p(currentP);

    // Make a half step for momentum at the beginning.
    p = p - epsilon * gradient / 2;
    // Alternate full steps for position and momentum
    for (int i = 0; i < l; ++i) {
        // Make a full step for the position
        beta = beta + epsilon * betaFunctor.fisherLLT().solve(p);
        betaFunctor.setBeta(beta);
        betaFunctor.gradient(gradient);
        if (i < l - 1) {
            // Make a full step for the momentum, except at end of trajectory
            p = p - epsilon * gradient;
        }
    }
    // Make a half step for momentum at the end.
    p = p - epsilon * gradient / 2;
    // Negate momentum at end of trajectory to make the proposal symmetric
    p = -p;

    // NOTE(mgnb): safe to do this because the fisher matrix is constant
    Eigen::VectorXd z = betaFunctor.fisherLLT().matrixL().solve(p);

    double currentK = 0.5 * currentZ.array().square().sum();
    double proposedU = betaFunctor.value();
    double proposedK = 0.5 * z.array().square().sum();

    if (randUniform(rng) < std::exp(currentU - proposedU + currentK - proposedK)) {
        return beta;
    }
    return betaCurrent;
}

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_HMC_HPP_
