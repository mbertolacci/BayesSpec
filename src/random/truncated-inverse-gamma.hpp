#ifndef SRC_RANDOM_TRUNCATED_INVERSE_GAMMA_HPP_
#define SRC_RANDOM_TRUNCATED_INVERSE_GAMMA_HPP_

#include <random>
#include "utils.hpp"

namespace bayesspec {

class TruncatedInverseGammaDistribution {
public:
    typedef double result_type;

    TruncatedInverseGammaDistribution(double alpha, double beta, double upper)
        : alpha_(alpha),
          beta_(beta) {
        logConst1_ = R::pgamma(
            1 / upper,
            alpha_,
            1 / beta_,
            // Gets upper tail probability
            0,
            // Gets log probability
            1
        );
    }

    template<class RNG>
    double operator()(RNG& rng) {
        double u = randUniform(rng);
        double logConst2 = std::log(u) + logConst1_;
        return 1 / R::qgamma(
            logConst2,
            alpha_,
            1 / beta_,
            // Specifies first argument is upper tail probability
            0,
            // Specifies first argument is log probability
            1
        );
    }

private:
    double alpha_;
    double beta_;
    double logConst1_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_TRUNCATED_INVERSE_GAMMA_HPP_
