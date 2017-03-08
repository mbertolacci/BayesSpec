#ifndef SRC_RNG_HPP_
#define SRC_RNG_HPP_

#include <RcppEigen.h>

#include <cstdint>
#include <random>

namespace bayesspec {

class RNG {
 public:
    RNG() {
        engine_ = std::mt19937_64(0);
    }

    explicit RNG(uint_fast64_t seed) {
        engine_ = std::mt19937_64(seed);
    }

    int randint(int lower, int upper) {
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(engine_);
    }

    double randu() {
        return uniformDistribution_(engine_);
    }

    double randn() {
        return normalDistribution_(engine_);
    }

    Eigen::VectorXd randn(unsigned int n) {
        Eigen::VectorXd x(n);
        for (unsigned int i = 0; i < n; ++i) {
            x[i] = randn();
        }
        return x;
    }

    double rande() {
        return exponentialDistribution_(engine_);
    }

    double randg(double alpha, double beta) {
        std::gamma_distribution<double> distribution(alpha, beta);
        return distribution(engine_);
    }

    static void initialise();

    std::mt19937_64 engine_;

 private:
    std::uniform_real_distribution<double> uniformDistribution_;  // U(0, 1)
    std::normal_distribution<double> normalDistribution_;  // N(0, 1)
    std::exponential_distribution<double> exponentialDistribution_;  // exp(1)
};

#if defined(__clang__)
extern RNG rng;
#else
extern thread_local RNG rng;
#endif

}  // namespace bayesspec

#endif  // SRC_RNG_HPP_
