#ifndef SRC_RANDOM_TRUNCATED_NORMAL_HPP_
#define SRC_RANDOM_TRUNCATED_NORMAL_HPP_

#include <random>

namespace bayesspec {

class TruncatedNormalDistribution {
public:
    typedef double result_type;

    TruncatedNormalDistribution(double mean, double stddev, double lower, double upper)
        : base_(mean, stddev),
          lower_(lower),
          upper_(upper) {}

    template<class RNG>
    double operator()(RNG& rng) {
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            double output = base_(rng);
            if (output > lower_ && output < upper_) {
                return output;
            }
        }
        throw std::runtime_error("Max iterations exceeded");
    }

private:
    const int MAX_ATTEMPTS = 10000;
    std::normal_distribution<double> base_;
    double lower_;
    double upper_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_TRUNCATED_NORMAL_HPP_
