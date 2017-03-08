#ifndef SRC_RANDOM_DIRICHLET_HPP_
#define SRC_RANDOM_DIRICHLET_HPP_

#include <RcppEigen.h>

namespace bayesspec {

class DirichletDistribution {
public:
    DirichletDistribution(const Eigen::VectorXd& alpha)
        : alpha_(alpha) {}

    template<class RNG>
    Eigen::VectorXd operator()(RNG& rng) {
        Eigen::VectorXd output(alpha_.size());

        for (unsigned int i = 0; i < alpha_.size(); ++i) {
            std::gamma_distribution<double> distribution(alpha_[i], 1);
            output[i] = distribution(rng);
        }
        output /= output.sum();
        return output;
    }

private:
    Eigen::VectorXd alpha_;
};

}  // namespace bayesspec

#endif  // SRC_RANDOM_DIRICHLET_HPP_
