#ifndef SRC_RANDOM_LOG_GAMMA_HPP_
#define SRC_RANDOM_LOG_GAMMA_HPP_

#include <random>

namespace bayesspec {

class LogGammaDistribution {
public:
    LogGammaDistribution(double shape, double logScale)
        : shape_(shape),
          logScale_(logScale),
          lambda_(1 / shape - 1),
          w_(shape / (E1_ * (1.0 - shape))),
          r_(1 / (1 + w_)),
          uniform_(),
          gamma_(shape, 1) {}

    template<class RNG>
    double operator()(RNG& rng) {
        if (shape_ > 0.2) {
            // Not a small shape; take the log of a gamma from <random>
            return logScale_ + std::log(gamma_(rng));
        }

        // Accept reject as per Liu, Martin and Syring,
        // https://arxiv.org/pdf/1302.1884.pdf
        for (int i = 0; i < MAX_ITERATIONS_; ++i) {
            double U = uniform_(rng);
            double z;
            if (U <= r_) {
                z = -std::log(U / r_);
            } else {
                z = std::log(uniform_(rng)) / lambda_;
            }
            double h = std::exp(-z - std::exp(-z / shape_));
            double eta;
            if (z >= 0) {
                eta = std::exp(-z);
            } else {
                eta = w_ * lambda_ * std::exp(lambda_ * z);
            }

            if (h / eta > uniform_(rng)) {
                return logScale_ - z / shape_;
            }
        }

        throw std::runtime_error("Max iterations exceeded");
    }

private:
    // Constants
    const double E1_ = std::exp(1);
    const int MAX_ITERATIONS_ = 10000;

    // Parameters
    double shape_;
    double logScale_;

    // Functions of parameters
    double lambda_;
    double w_;
    double r_;

    // Distributions
    std::uniform_real_distribution<double> uniform_;
    std::gamma_distribution<double> gamma_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_LOG_GAMMA_HPP_
