#ifndef SRC_RANDOM_INVERSE_GAMMA_HPP_
#define SRC_RANDOM_INVERSE_GAMMA_HPP_

#include <random>

namespace bayesspec {

class InverseGammaDistribution {
public:
    typedef double result_type;

    InverseGammaDistribution(double alpha, double beta)
        : gamma_(alpha, 1),
          beta_(beta) {}

    template<class RNG>
    double operator()(RNG& rng) {
        return beta_ / gamma_(rng);
    }

private:
    std::gamma_distribution<double> gamma_;
    double beta_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_INVERSE_GAMMA_HPP_
