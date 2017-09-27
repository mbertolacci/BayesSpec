#ifndef SRC_RANDOM_POLYAGAMMA_HPP_
#define SRC_RANDOM_POLYAGAMMA_HPP_

#include <RcppEigen.h>
#include "utils.hpp"

namespace bayesspec {

class PolyagammaDistribution {
public:
    explicit PolyagammaDistribution(double z) {
        z_ = std::fabs(z) / 2;

        K_ = Eigen::numext::abs2(M_PI) / 8 + Eigen::numext::abs2(z_) / 2;
        p_ = M_PI / (2 * K_) * std::exp(-K_ * POLYAGAMMA_TRUNC);
        q_ = 2 * std::exp(-z_) * pInverseTruncatedGauss(POLYAGAMMA_TRUNC, z_, 1.0);
    }

    template<typename RNG>
    double operator()(RNG& rng) {
        if (std::isnan(z_)) {
            throw std::runtime_error("PolyagammaDistribution given NaN");
        }
        return samplePolyagammaSingle(rng);
    }

private:
    double z_;
    double K_;
    double p_;
    double q_;

    const double POLYAGAMMA_TRUNC = 0.2;

    double pnorm(double value) {
        return 0.5 * std::erfc(-value * M_SQRT1_2);
    }

    double pInverseTruncatedGauss(double x, double z, double lambda) {
        return (
            pnorm(std::sqrt(lambda / x) * (x * z - 1))
            + std::exp(2 * lambda * z) * pnorm(-std::sqrt(lambda / x) * (x * z + 1))
        );
    }

    double polyagammaCoeff(double x, unsigned int n) {
        double nPlusHalf = static_cast<double>(n) + 0.5;

        if (x <= POLYAGAMMA_TRUNC) {
            double twoOverPiX = 2 / (M_PI * x);
            return (
                M_PI * nPlusHalf * twoOverPiX * std::sqrt(twoOverPiX) * std::exp(
                    -2 * Eigen::numext::abs2(nPlusHalf) / x
                )
            );
        } else {
            return (
                M_PI * nPlusHalf * std::exp(
                    -Eigen::numext::abs2(nPlusHalf) * Eigen::numext::abs2(M_PI) * x / 2
                )
            );
        }
    }

    template<typename RNG>
    double sampleInverseTruncatedGauss(RNG& rng) {
        double X, U;

        if (z_ < 1 / POLYAGAMMA_TRUNC) {
            double alpha, E1, E2;

            do {
                do {
                    E1 = randExponential(rng);
                    E2 = randExponential(rng);
                } while (Eigen::numext::abs2(E1) > 2 * E2 / POLYAGAMMA_TRUNC);

                X = POLYAGAMMA_TRUNC / ((1 + POLYAGAMMA_TRUNC * E1) * (1 + POLYAGAMMA_TRUNC * E1));

                alpha = std::exp(-X * Eigen::numext::abs2(z_) / 2);

                U = randUniform(rng);
            } while (U > alpha);
        } else {
            double mu = 1 / z_;
            do {
                double Y = Eigen::numext::abs2(randNormal(rng));
                X = mu + 0.5 * Eigen::numext::abs2(mu) * Y - 0.5 * mu * std::sqrt(4 * mu * Y + Eigen::numext::abs2(mu * Y));

                U = randUniform(rng);
                if (U > mu / (mu + X)) {
                    X = Eigen::numext::abs2(mu) / X;
                }
            } while (X > POLYAGAMMA_TRUNC);
        }

        return X;
    }

    template<typename RNG>
    double samplePolyagammaSingle(RNG& rng) {
        while (true) {
            double X;

            if (randUniform(rng) < p_ / (p_ + q_)) {
                // Truncated exponential
                X = POLYAGAMMA_TRUNC + randExponential(rng) / K_;
            } else {
                // Truncated inverse Gaussian
                X = sampleInverseTruncatedGauss(rng);
            }

            unsigned int n = 0;
            double S = polyagammaCoeff(X, 0);
            double Y = randUniform(rng) * S;

            while (true) {
                ++n;
                if (n % 2 == 1) {
                    S -= polyagammaCoeff(X, n);
                    if (Y <= S) {
                        return X / 4;
                    }
                } else {
                    S += polyagammaCoeff(X, n);
                    if (Y > S) {
                        break;
                    }
                }
            }
        }

        return 0;
    }
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_POLYAGAMMA_HPP_
