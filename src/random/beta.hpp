#ifndef SRC_RANDOM_BETA_HPP_
#define SRC_RANDOM_BETA_HPP_

#include <RcppEigen.h>

namespace bayesspec {

class BetaDistribution {
public:
    BetaDistribution(double a, double b)
        : a_(a),
          b_(b) {}

    template<class RNG>
    double operator()(RNG& rng) {
        double x = std::gamma_distribution<double>(a_, 1)(rng);
        double y = std::gamma_distribution<double>(b_, 1)(rng);
        return x / (x + y);
    }

private:
    double a_;
    double b_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_BETA_HPP_
