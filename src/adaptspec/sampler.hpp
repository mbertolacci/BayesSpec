#ifndef SRC_ADAPTSPEC_SAMPLER_HPP_
#define SRC_ADAPTSPEC_SAMPLER_HPP_

#include <RcppEigen.h>

#include "prior.hpp"
#include "state.hpp"

namespace bayesspec {

class AdaptSpecSampler {
public:
    AdaptSpecSampler(
        const Eigen::MatrixXd& x,
        const AdaptSpecParameters& start,
        double probMM1,
        const AdaptSpecPrior& prior
    ) : current_(start, x, prior, probMM1) {}

    template<typename RNG>
    void sample(RNG& rng) {
        current_.sample(rng);
    }

    const AdaptSpecParameters& getCurrent() const {
        return current_.parameters;
    }

private:
    AdaptSpecState current_;
};

}  // namespace bayespec

#endif  // SRC_ADAPTSPEC_SAMPLER_HPP_