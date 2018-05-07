#ifndef SRC_RANDOM_LOG_BETA_HPP_
#define SRC_RANDOM_LOG_BETA_HPP_

#include <RcppEigen.h>
#include "log-gamma.hpp"

namespace bayesspec {

class LogBetaDistribution {
public:
    LogBetaDistribution(double a, double b)
        : lgA_(a, 0),
          lgB_(b, 0) {}

    template<class RNG>
    double operator()(RNG& rng) {
        double logX = lgA_(rng);
        double logY = lgB_(rng);
        double m = std::max(logX, logY);
        return logX - std::log(std::exp(logX - m) + std::exp(logY - m)) - m;
    }

private:
    LogGammaDistribution lgA_;
    LogGammaDistribution lgB_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_LOG_BETA_HPP_
