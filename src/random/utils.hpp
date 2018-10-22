#ifndef SRC_RANDOM_UTILS_HPP_
#define SRC_RANDOM_UTILS_HPP_

#include <random>

namespace bayesspec {

template<typename RNG>
int randInteger(int lower, int upper, RNG& rng) {
    return std::uniform_int_distribution<int>(lower, upper)(rng);
}

template<typename RNG>
double randUniform(RNG& rng) {
    return std::uniform_real_distribution<double>()(rng);
}

template<typename RNG>
double randExponential(RNG& rng) {
    return std::exponential_distribution<double>()(rng);
}

template<typename RNG>
double randGamma(double shape, double scale, RNG& rng) {
    return std::gamma_distribution<double>(shape, scale)(rng);
}

template<typename RNG>
double randNormal(RNG& rng) {
    return std::normal_distribution<double>()(rng);
}

template<typename RNG>
Eigen::VectorXd randNormal(unsigned int n, RNG& rng) {
    std::normal_distribution<double> distribution;
    Eigen::VectorXd output(n);
    for (unsigned int i = 0; i < n; ++i) {
        output[i] = distribution(rng);
    }
    return output;
}

template<typename RNG, typename T>
T randElement(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& elements,
    const Eigen::VectorXd& weights,
    RNG& rng
) {
    double u = randUniform(rng) * weights.sum();
    for (unsigned int i = 0; i < elements.size(); ++i) {
        u -= weights[i];
        if (u <= 0) {
            return elements[i];
        }
    }
    return elements[elements.size() - 1];
}

}  // namespace bayesspec

#endif  // SRC_RANDOM_UTILS_HPP_
