#ifndef SRC_RANDOM_TRUNCATED_NORMAL_HPP_
#define SRC_RANDOM_TRUNCATED_NORMAL_HPP_

#include <random>

namespace bayesspec {

class TruncatedNormalDistribution {
public:
    typedef double result_type;

    TruncatedNormalDistribution(double mean, double stddev, double lower, double upper)
        : mean_(mean),
          stddev_(stddev),
          standardLower_((lower - mean) / stddev),
          standardUpper_((upper - mean) / stddev),
          unif_(),
          standardNormal_() {}

    template<class RNG>
    double operator()(RNG& rng) {
        double standardResult;
        if (standardLower_ > THRESHOLD) {
            standardResult = devroye(rng, standardLower_, standardUpper_);
        } else if (standardUpper_ < -THRESHOLD) {
            standardResult = -devroye(rng, -standardUpper_, -standardLower_);
        } else {
            standardResult = rejection(rng);
        }
        return mean_ + stddev_ * standardResult;
    }

    template<class RNG>
    double devroye(RNG& rng, double lower, double upper) {
        double c = lower * lower / 2;
        double f = std::expm1(c - upper * upper / 2);

        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            double output = c - std::log(1 + unif_(rng) * f);
            double u = unif_(rng);

            if (u * u * output < c) {
                return std::sqrt(2 * output);
            }
        }

        throw std::runtime_error("Max iterations exceeded in devroye");
    }

    template<class RNG>
    double rejection(RNG& rng) {
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            double output = standardNormal_(rng);
            if (output > standardLower_ && output < standardUpper_) {
                return output;
            }
        }

        throw std::runtime_error("Max iterations exceeded in rejection");
    }

private:
    const int MAX_ATTEMPTS = 10000;
    const double THRESHOLD = 0.4;

    double mean_;
    double stddev_;
    double standardLower_;
    double standardUpper_;

    std::uniform_real_distribution<double> unif_;
    std::normal_distribution<double> standardNormal_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_TRUNCATED_NORMAL_HPP_
